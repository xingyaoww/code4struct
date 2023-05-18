"""
Augment processed ACE data with in-context examples.
"""
import os
import pathlib
import argparse
import json
import copy

from collections import defaultdict
from typing import Mapping, List, Dict
from tqdm import tqdm


def process_single_event(
    ex,
    name_to_ontology: Mapping[str, dict],
    line_idx: int,
    output_code_dir: str,
    hierarchy_split: dict,
    k_shot_examples: List[dict],
    fwrite,
    ace_roles_full_context: str,
    args
):
    # skip examples without event mentions
    # data = copy.deepcopy(ex)
    parent_cls_name = ex["parent_cls_name"]
    cur_cls_name = ex["cur_cls_name"]
    evt_type = parent_cls_name + ":" + cur_cls_name.replace("_", "-")

    # 1. Process hierarchy
    cur_hierarchy = hierarchy_split[ex["parent_cls_name"]]
    cur_hierarchy["train"] = cur_hierarchy["train"].replace(
        "-", "_")  # normalize
    cur_hierarchy["test"] = list(
        map(lambda s: s.replace("-", "_"), cur_hierarchy["test"]))
    # We do not augment examples for event types used as training data source
    if cur_cls_name == cur_hierarchy["train"]:
        # print(f"Skipping {evt_type} as it is used as training data source.")
        return
    assert cur_cls_name in cur_hierarchy['test'], f"{cur_cls_name} not in {cur_hierarchy['test']}"
    # use sibling data as in-context examples
    sibling_cls_name = cur_hierarchy["train"]

    # 2. Augment code context and examples
    cur_event_ontology = name_to_ontology[evt_type]

    # Load in-context examples for currrent class
    if args.only_sibling_examples:
        incontext_examples = []
    else:
        incontext_examples = copy.deepcopy(k_shot_examples[cur_cls_name])
        if len(incontext_examples) >= args.n_hierarchy_incontext_examples and not args.hierarchy_augment_nearest_neighbor:
            # if we have enough examples (as the baseline experiment), we don't need to infer them again
            return
    assert len(
        incontext_examples) < args.n_hierarchy_incontext_examples or args.hierarchy_augment_nearest_neighbor

    # 2.1 Update add sibling class definition to code_context
    # insert sibling_event_ontology["code"] before "class {cur_cls_name}({parent_cls_name}):"
    sibling_event_ontology = name_to_ontology[parent_cls_name +
                                              ":" + sibling_cls_name.replace("_", "-")]
    code_context = ex["code_context"]
    code_context = code_context.replace(
        f"class {cur_cls_name}({parent_cls_name}):",
        sibling_event_ontology["code"] +
        f"class {cur_cls_name}({parent_cls_name}):"
    )

    # 2.2 Add in-context examples from parent class
    sibling_incontext_examples = copy.deepcopy(
        k_shot_examples[sibling_cls_name])
    if args.hierarchy_augment_nearest_neighbor:
        assert "embedding" in incontext_examples[0]
        cur_embedding = SENT_MODEL.encode(
            ex["sentence"],
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        incontext_examples = list(reversed(sorted(
            incontext_examples + sibling_incontext_examples,
            key=lambda _train_ex: util.pytorch_cos_sim(
                cur_embedding, _train_ex["embedding"]
            ).item(),
            reverse=True
        )[:args.n_hierarchy_incontext_examples]))

    else:
        incontext_examples = sibling_incontext_examples[
            :args.n_hierarchy_incontext_examples - len(incontext_examples)
        ] + incontext_examples
    # else:
    # incontext_examples = incontext_examples[:args.n_hierarchy_incontext_examples]
    assert len(
        incontext_examples) == args.n_hierarchy_incontext_examples, f"{len(incontext_examples)} != {args.n_hierarchy_incontext_examples}"

    # 3. Update data
    data = copy.deepcopy(ex)
    line_idx = data["line_idx"]
    event_idx = data["event_idx"]
    doc_id = data["doc_id"]
    sent_id = data["sent_id"]
    data["code_context"] = code_context
    # filter out "embedding" field for each dict in incontext_examples
    data["incontext_examples"] = list(map(
        lambda _ex: {k: v for k, v in _ex.items() if k != "embedding"}, incontext_examples))
    # full_prompt = code_context + text_prompt + instantiation_prompt
    data["full_prompt"] = data["code_context"] + \
        data["text_prompt"] + data["instantiation_prompt"]
    assert len(data["incontext_examples"]) == args.n_hierarchy_incontext_examples, \
        f"len(data['incontext_examples']) = {len(data['incontext_examples'])}"

    # 4. Save data and code to file
    fwrite.write(json.dumps(data) + "\n")
    with open(os.path.join(output_code_dir, f"{line_idx}_{event_idx}.py"), "w") as f:
        f.write(f"# Doc ID: {doc_id} \n# Sent ID: {sent_id}\n")
        f.write(data["full_prompt"])

    with open(os.path.join(output_code_dir, f"{line_idx}_{event_idx}_incontext.py"), "w") as f:
        f.write(f"# Doc ID: {doc_id} \n# Sent ID: {sent_id}\n")
        f.write(data["code_context"])
        for _train_example in incontext_examples:
            f.write(
                _train_example["text_prompt"] +
                _train_example["gt_instance_code"] + "\n"
            )
        f.write(data["text_prompt"] + data["instantiation_prompt"])

    with open(os.path.join(output_code_dir, f"{line_idx}_{event_idx}_gt.py"), "w") as f:
        f.write(f"# Doc ID: {doc_id} \n# Sent ID: {sent_id}\n")
        f.write(data["code_context"] + data["text_prompt"])
        f.write(data["gt_instance_code"])
    with open(os.path.join(output_code_dir, f"{line_idx}_{event_idx}.json"), "w") as f:
        f.write(json.dumps(data, indent=4) + "\n")


def main(args):
    # 1. Load ontology
    with open(args.parsed_ace_roles) as f:
        parsed_ace_roles = json.load(f)
    name_to_ontology = {parsed["name"]: parsed for parsed in parsed_ace_roles}
    print("Ontology loaded.")

    with open(".".join(args.parsed_ace_roles.split(".")[:-1]) + ".py") as f:
        ace_roles_full_context = f.read()

    if args.n_hierarchy_incontext_examples > 0:
        with open(args.hierarchy_split_filepath, "r") as f:
            hierarchy_split = json.load(f)

        def load_in_context_examples(train_path, k_shot: int):
            def _covert_cur_cls_name_to_key(cur_cls_name) -> str:
                if isinstance(cur_cls_name, list):
                    cur_cls_name = "+".join(sorted(cur_cls_name))
                    assert isinstance(cur_cls_name, str), cur_cls_name
                    return cur_cls_name

                assert isinstance(cur_cls_name, str)
                return cur_cls_name
            # key: cur_cls_name, value: list of examples for that event type
            k_shot_examples: Mapping[str, List[Dict]] = defaultdict(list)
            assert k_shot >= 0
            if k_shot > 0:
                with open(train_path, "r") as fread:
                    _train_examples = [json.loads(line)
                                       for line in tqdm(fread)]
                    print(
                        f"Loaded {len(_train_examples)} training examples for in-context completion.")
                    for _train_example in tqdm(_train_examples):
                        if args.hierarchy_augment_nearest_neighbor:
                            _train_example["embedding"] = SENT_MODEL.encode(
                                _train_example["sentence"],
                                convert_to_tensor=True,
                                show_progress_bar=False,
                            )
                        # will be loaded in the order of the input file
                        k_shot_examples[
                            _covert_cur_cls_name_to_key(
                                _train_example["cur_cls_name"])
                        ].append(_train_example)

            # filter down to k_shot
            n_total = 0
            for k, v in k_shot_examples.items():
                k_shot_examples[k] = v[:k_shot]
                n_total += len(k_shot_examples[k])
            print(f"Total {n_total} in-context examples.")
            return k_shot_examples

        k_shot_examples = load_in_context_examples(
            args.input_train_filepath, args.n_hierarchy_incontext_examples)
    else:
        raise NotImplementedError

    # 2. Figure out output filepaths
    subset = args.input_filepath.split("/")[-1].split(".")[0]
    pathlib.Path(args.output_filedir).mkdir(parents=True, exist_ok=True)
    output_filepath = os.path.join(args.output_filedir, subset + ".jsonl")
    output_code_dir = os.path.join(args.output_filedir, subset)
    pathlib.Path(output_code_dir).mkdir(parents=True, exist_ok=True)
    print(
        f"Writing mapping (.json) to {output_filepath}, code (.py) to {output_code_dir}")

    # 3. Process input data file (i.e. subset) to output
    with open(args.input_filepath, "r") as fread, open(output_filepath, "w") as fwrite:
        for line_idx, line in tqdm(enumerate(fread)):
            ex = json.loads(line)

            process_single_event(
                ex,
                name_to_ontology,
                line_idx,
                output_code_dir,
                hierarchy_split,
                k_shot_examples,
                fwrite,
                ace_roles_full_context,
                args
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parsed-ace-roles', required=True)
    parser.add_argument('--input-filepath', required=True)
    parser.add_argument('--input-train-filepath', required=True)
    parser.add_argument('--output-filedir', required=True)

    parser.add_argument('--hierarchy-split-filepath')
    parser.add_argument('--n-hierarchy-incontext-examples',
                        default=0, type=int)
    parser.add_argument(
        '--hierarchy-augment-nearest-neighbor', action='store_true')
    parser.add_argument('--only-sibling-examples', action='store_true')
    args = parser.parse_args()

    if args.n_hierarchy_incontext_examples > 0:
        assert os.path.exists(
            args.hierarchy_split_filepath), "Hierarchy split file not found."
        if args.hierarchy_augment_nearest_neighbor:
            from sentence_transformers import SentenceTransformer, util
            SENT_MODEL = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2')
            print(
                f"Using {args.n_hierarchy_incontext_examples} nearest neighbor to augment hierarchy examples.")
        else:
            print(
                f"Using {args.n_hierarchy_incontext_examples} in order to fill in missing context.")

    main(args)
