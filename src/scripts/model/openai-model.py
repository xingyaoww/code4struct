import os
import time
import json
import pathlib
import openai
import openai.error
import argparse
import logging
from tqdm import tqdm
from collections import defaultdict, deque
from typing import List, Dict, Mapping

from src.models import (
    ModelInferenceLoop,
    add_model_args_to_parser,
    get_prompt
)


class CodexInferenceLoop(ModelInferenceLoop):
    def prompt_inference_func(
        self,
        prompts: List[str],
        examples: List[dict],
    ):
        try:
            response = openai.Completion.create(
                model=self.args.model,
                prompt=prompts,
                temperature=self.args.temperature,
                max_tokens=self.args.max_tokens,
                top_p=self.args.top_p,
                n=self.args.n_generations,
                logprobs=self.args.logprobs,
                frequency_penalty=0,
                presence_penalty=0,
                stop=self.args.stop_sequences
            )
        except openai.error.RateLimitError as e:
            self.logger.info("Rate limit error, sleeping for 10 secs.")
            # logging.exception(e)
            time.sleep(10)
            # to resume in the next iteration
            self.input_examples.appendleft(examples)
            return

        assert len(response["choices"]) == len(
            examples) * self.args.n_generations
        for example_id, cur_example in enumerate(examples):
            # figure out choices for this example
            choices_for_example = response["choices"][
                example_id * self.args.n_generations: (example_id+1) * self.args.n_generations
            ]
            cur_response = copy.deepcopy(response)
            cur_response["choices"] = choices_for_example

            # figure out the instance codes
            generated_instance_codes = [
                cur_example["instantiation_prompt"] + choice["text"]
                for choice in choices_for_example
            ]

            # Write the response to the output file
            data = {
                "input": cur_example,
                "output": cur_response,
                "prompt": prompts[example_id],
                "instance_code": generated_instance_codes
            }
            self.output_fwrite.write(json.dumps(data) + "\n")
        self.output_fwrite.flush()
        self.pbar.update(1)
        time.sleep(10)  # to avoid rate limit


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger("openai").setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='Codex')
    parser = add_model_args_to_parser(parser)
    parser.add_argument(
        '--logprobs', type=int, default=None,
        help='top `logprobs` to return'
    )
    parser.add_argument(
        '--model',
        type=str,
        default="code-davinci-002",
        help='model to use'
    )
    parser.add_argument(
        '--do-detection',
        action='store_true',
    )
    args = parser.parse_args()

    if args.do_detection:
        args.stop_sequences = [")"]
    else:
        args.stop_sequences = ["\"\"\"", "class", "print", "#"]

    openai.api_key = os.getenv("OPENAI_API_KEY")
    inference_loop = CodexInferenceLoop(args, logger)
    inference_loop.run()
