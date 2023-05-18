import os
import pathlib
import json

from sentence_transformers import SentenceTransformer, util
from collections import defaultdict, deque
from typing import Deque, List, Dict, Mapping
from tqdm import tqdm

SENT_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def _covert_cur_cls_name_to_key(cur_cls_name) -> str:
    if isinstance(cur_cls_name, list):
        cur_cls_name = "+".join(sorted(cur_cls_name))
        assert isinstance(cur_cls_name, str), cur_cls_name
        return cur_cls_name

    assert isinstance(cur_cls_name, str)
    return cur_cls_name


def _load_in_context_examples(args, logger) -> Mapping[str, List[Dict]]:
    # key: cur_cls_name, value: list of examples for that event type
    k_shot_examples: Mapping[str, List[Dict]] = defaultdict(list)
    assert args.k_shot >= 0
    if args.k_shot > 0:
        if args.input_train_filepath is None or not os.path.exists(args.input_train_filepath):
            logger.info(
                f"Cannot find in-context examples file at {args.input_train_filepath}"
            )
            return None
        with open(args.input_train_filepath, "r") as fread:
            _train_examples = [json.loads(line) for line in tqdm(fread)]
            logger.info(
                f"Loading {len(_train_examples)} training examples for in-context completion.")
            for _train_example in tqdm(_train_examples):
                if args.do_k_nearest_neighbors:
                    _train_example["embedding"] = SENT_MODEL.encode(
                        _train_example["sentence"],
                        convert_to_tensor=True,
                        show_progress_bar=False,
                    )

                # will be loaded in the order of the input file
                k_shot_examples[
                    _covert_cur_cls_name_to_key(_train_example["cur_cls_name"])
                ].append(
                    _train_example
                )
    return k_shot_examples


def get_prompt(
    cur_example: Dict,
    k_shot_examples: Mapping[str, List[Dict]],
    n_shots: int,
    do_k_nearest_neighbors: bool,
) -> str:
    assert n_shots >= 0
    if n_shots == 0:
        return cur_example["full_prompt"]

    if "incontext_examples" not in cur_example:
        train_examples = list(k_shot_examples.get(
            _covert_cur_cls_name_to_key(cur_example["cur_cls_name"])
        ))  # copy the list

        if do_k_nearest_neighbors and len(train_examples) > n_shots:
            assert "embedding" in train_examples[0]
            cur_embedding = SENT_MODEL.encode(
                cur_example["sentence"],
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            train_examples = sorted(
                train_examples,
                key=lambda ex: util.pytorch_cos_sim(
                    cur_embedding, ex["embedding"]
                ).item(),
                reverse=True
            )[:n_shots]

        else:
            train_examples = train_examples[:n_shots]
    else:
        train_examples = cur_example["incontext_examples"]

    if len(train_examples) < n_shots:
        print(
            f"Warning: only {len(train_examples)} in-context examples for {cur_example['cur_cls_name']}")
    assert len(train_examples) <= n_shots

    # code context contains event ontology
    prompt = cur_example["code_context"]
    # add in-context training examples
    for _train_example in train_examples:
        prompt += _train_example["text_prompt"] + \
            _train_example["gt_instance_code"] + "\n"
    # add current example for inference
    prompt += cur_example["text_prompt"] + cur_example["instantiation_prompt"]
    return prompt


def add_model_args_to_parser(parser):
    # e.g., test.jsonl
    parser.add_argument('--input-filepath')
    # training data for in-context completion
    parser.add_argument('--input-train-filepath')  # e.g., train.jsonl
    parser.add_argument('--output-dir')

    # Number of examples to use for in-context completion
    parser.add_argument('--k-shot', type=int, default=0)

    # Generation settings
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--top-p', type=int, default=1)
    parser.add_argument('--n-generations', type=int, default=1,
                        help='number of completions to generate')

    # In-context training settings
    parser.add_argument('--do-k-nearest-neighbors', action='store_true')

    parser.add_argument('--batch-size', type=int, default=1)
    return parser


class ModelInferenceLoop:

    def _load_input_examples_and_output_fhandle(self):
        def _get_example_id(ex):
            return f"{ex['line_idx']}-{ex['event_idx']}"

        # 1. Figure out the output filepath
        input_filename = os.path.basename(self.args.input_filepath)
        output_filepath = os.path.join(self.args.output_dir, input_filename)
        pathlib.Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Infer on {self.args.input_filepath}")
        self.logger.info(f"Write results to {output_filepath}")
        output_fwrite = open(output_filepath, 'a')
        with open(output_filepath + ".config.json", "w") as f:
            f.write(json.dumps(self.args.__dict__, indent=4) + "\n")

        # 2. Load the output file (if exists) to figure out which examples have been processed
        inferred_ids = set()
        if os.path.exists(output_filepath):
            with open(output_filepath, "r") as fread:
                output_results = [json.loads(line) for line in fread]
            inferred_ids.update(
                (_get_example_id(result["input"]) for result in output_results))
            self.logger.info(
                f"Found {len(output_results)} existing results on output file.")

        # 3. Load the input file AND filter out existing results
        with open(self.args.input_filepath, "r") as fread:
            input_examples = [json.loads(line) for line in fread]
            self.logger.info(
                f"Loaded {len(input_examples)} input examples for inference.")
        if len(inferred_ids) > 0:
            input_examples = [
                ex for ex in input_examples if _get_example_id(ex) not in inferred_ids]
            self.logger.info(f"Filtered to {len(input_examples)} examples.")

        return input_examples, output_fwrite

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        _input_examples, self.output_fwrite = self._load_input_examples_and_output_fhandle()

        # (Optional) Load the training data for in-context completion when k-shot > 0
        self.k_shot_examples = _load_in_context_examples(
            self.args, self.logger)

        # batch input examples
        assert self.args.batch_size > 0
        batched_input_examples = []
        for i in range(0, len(_input_examples), self.args.batch_size):
            batched_input_examples.append(
                _input_examples[i:i+self.args.batch_size])
        self.input_examples: Deque[List[dict]] = deque(batched_input_examples)
        self.logger.info(
            f"{self.args.batch_size} examples per batch. Total {len(self.input_examples)} batches.")

        self.pbar = tqdm(total=len(self.input_examples))
        self.logger.info(f"Model Inference Loop Initialization Complete.")

    def prompt_inference_func(
        self,
        prompts: List[str],
        examples: List[dict]
    ):
        """
        (1) Inference the prompt
        (2) save results to self.output_fwrite
        (3) Update self.pbar
        """
        raise NotImplementedError

    def run(self):
        while len(self.input_examples) > 0:
            examples: List[dict] = self.input_examples.popleft()
            prompts: List[str] = [
                get_prompt(
                    example,
                    self.k_shot_examples,
                    self.args.k_shot,
                    self.args.do_k_nearest_neighbors,
                ) for example in examples
            ]

            # Call the prompt inference function
            # result should be write to output_fwrite by the prompt_inference_func
            _ = self.prompt_inference_func(
                prompts,
                examples,
            )

        self.pbar.close()
        self.output_fwrite.close()
