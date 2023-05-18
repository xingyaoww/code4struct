"""
Convert ACE05 dataset to code generation ready format, using previously converted ace roles.

python3 src/scripts/data/process_ACE_dataset.py \
    --parsed-ace-roles data/ontology/ace/processed/ace_roles.json \
    --input-filepath data/extraction/ace/pro_mttrig_id/json/test.oneie.json \
    --output-filedir data/extraction/ace/processed
"""
import os
import pathlib
import argparse
import json
import amrlib

from collections import defaultdict
from typing import Mapping, List
from tqdm import tqdm

from src.scripts.data.build_ldc_base_entities import entity_type_mapping


def _sort_unique_roles(roles: List[str]) -> List[str]:
    return sorted(list(set(roles)), key=lambda s: s.lower())


def build_prompt(
        cur_cls_name: str,
        sentence: str,
        event_ontology: dict,
        args) -> str:
    if args.event_detection:
        sentence = sentence.replace("\"", "'")
        text_prompt = (
            "def assert_event_trigger_words_and_type(event_text, trigger_words: List[str], event_type):\n"
            "    # trigger word need to be a word in the original sentence\n"
            "    for word in trigger_words:\n"
            "        assert word in event_text.split()\n"
            "    event = convert_text_to_event(event_text)\n"
            "    assert event.trigger_words == trigger_words\n"
            "    assert isinstance(event, event_type)\n"
        )
        instantiation_prompt = f"\nassert_event_trigger_words_and_type(\n    \"{sentence}\", "
        return text_prompt, instantiation_prompt

    # Pure text for GPT-3
    if args.pure_text_prompt:
        if args.mark_trigger:
            text_prompt = (
                f"Translate the following sentence into an instance of {cur_cls_name} event. "
                "The trigger word(s) of the event is marked with **trigger word**.\n"
                f"\"{sentence}\"\n"
            )
        else:
            text_prompt = f"Translate the following sentence into an instance of {cur_cls_name} event:\n{sentence}\n"
        # enforce sorted order for attributes/roles in prompt
        instantiation_prompt = f"1. {_sort_unique_roles(event_ontology['event']['roles'])[0].lower()}: ("
        return text_prompt, instantiation_prompt

    # Code for Codex
    if args.predict_event_type:
        text_prompt = (
            f"\"\"\"\nTranslate the following sentences to event(s):\n"
            f"\"{sentence}\"\n"
            f"\"\"\"\n"
        )
        instantiation_prompt = f"events: List[Event] = [\n"
        return text_prompt, instantiation_prompt

    # Common case for Event Argument Extraction
    if args.reduce_hallucination:
        text_prompt = (
            f"\"\"\"\nTranslate the following sentence into an instance of {cur_cls_name}. "
            "Only use information that can be founded in the text as arguments. "
            "Use [] as arguments when no information about them is presented in the text. \n"
            f"\"{sentence}\"\n"
            # f"\"\"\"\n"
        )
    elif args.mark_trigger:
        text_prompt = (
            f"\"\"\"\nTranslate the following sentence into an instance of {cur_cls_name}. "
            f"The trigger word(s) of the event is marked with **trigger word**.\n"
            f"\"{sentence}\"\n"
            # f"\"\"\"\n"
        )
    else:
        text_prompt = (
            f"\"\"\"\nTranslate the following sentence into an instance of {cur_cls_name}:\n"
            f"\"{sentence}\"\n"
            # f"\"\"\"\n"
        )

    if args.add_amr:
        global AMR_STOG
        amr_graphs = AMR_STOG.parse_sents([sentence])
        assert len(amr_graphs) == 1
        amr_graph = amr_graphs[0]
        # filter out comments
        amr_str = "\n".join(filter(
            lambda s: not s.startswith("#"),
            amr_graph.splitlines()
        ))
        text_prompt += f"\nAbstract Meaning Representation of the given sentence:\n{amr_str}\n\"\"\"\n"
    else:
        text_prompt += "\"\"\"\n"

    instantiation_prompt = f"{cur_cls_name.lower()}_event = {cur_cls_name}(\n"
    return text_prompt, instantiation_prompt


def build_gt_instance_code(
    cur_cls_name: str,
    cur_event_mentions: dict,
    entity_mentions: dict,
    event_ontology: dict,
    instantiation_prompt: str,
    define_event_var: bool = True,
    args=None,
) -> str:
    if args.event_detection:
        gt_code = instantiation_prompt
        gt_code += f"\n    [\"{cur_event_mentions['trigger']['text']}\"], "
        gt_code += f"\n    {cur_cls_name}\n)"
        return gt_code

    entity_id_to_mentions = {m["id"]: m for m in entity_mentions}

    # 1. extract arguments with entity types
    attr_to_args = defaultdict(list)
    for argument in cur_event_mentions['arguments']:
        entity_metion = entity_id_to_mentions[argument['entity_id']]
        entity_type = entity_metion['entity_type']
        if args.informative_entity_type:
            entity_type = entity_type_mapping[entity_type]
        attr_to_args[argument['role'].lower()].append(
            (entity_type, argument['text'])
        )

    # 2. build code
    if args.pure_text_prompt:
        gt_code = ""
        # enforce sorted order for attributes/roles in prompt
        for i, attr in enumerate(_sort_unique_roles(event_ontology['event']['roles'])):
            attr = attr.lower()
            arguments = attr_to_args.get(attr, None)
            if arguments is None:
                gt_code += f"{i+1}. {attr}: () \"\"\n"
            else:
                gt_code += f"{i+1}. {attr}: "
                gt_code += ", ".join(
                    [f"({arg[0]}) \"{arg[1]}\"" for arg in arguments])
                gt_code += "\n"
        return gt_code

    if define_event_var:
        gt_code = f"{cur_cls_name.lower()}_event = {cur_cls_name}(\n"
    else:
        gt_code = f"{cur_cls_name}(\n"

    for attr, arguments in attr_to_args.items():
        gt_code += f"    {attr}=[\n"
        for arg in arguments:
            gt_code += f"        {arg[0]}(\"{arg[1]}\"),\n"
        gt_code += f"    ],\n"
    gt_code += f")\n"
    return gt_code


def mark_trigger_in_sentence(
    tokens: List[str],
    gt_event: dict,
) -> str:
    # copy tokens
    tokens = tokens.copy()
    trigger = gt_event["trigger"]
    start, end = trigger["start"], trigger["end"]
    tokens[start] = f"**{tokens[start]}"
    tokens[end-1] = f"{tokens[end-1]}**"
    return " ".join(tokens)


def process_single_event(
    ex,
    name_to_ontology: Mapping[str, dict],
    line_idx: int,
    output_code_dir: str,
    fwrite,
    ace_roles_full_context: str,
    args
):
    # skip examples without event mentions
    for event_idx in range(len(ex['event_mentions'])):
        evt_type = ex['event_mentions'][event_idx]['event_type']
        if evt_type not in name_to_ontology:  # skip rare event type
            print(f"Rare event type: {evt_type}")
            continue

        # 3.1 Build code prompt from sentence and event template
        event_ontology = name_to_ontology[evt_type]
        gt_event = ex['event_mentions'][event_idx]
        cur_cls_name = event_ontology["cur_cls_name"]
        doc_id = ex["doc_id"]
        sent_id = ex["sent_id"]

        if args.event_detection:
            # predict event type
            code_context = ace_roles_full_context
        else:
            code_context = event_ontology["code_with_context"]

        # Get the sentence with trigger marked (if needed)
        if args.mark_trigger:
            sentence = mark_trigger_in_sentence(
                ex["tokens"], gt_event
            )
        else:
            sentence = ex["sentence"]
        text_prompt, instantiation_prompt = build_prompt(
            cur_cls_name, sentence, event_ontology, args)
        full_prompt = code_context + text_prompt + instantiation_prompt

        # 3.2 Build groundtruth code (for in-context learning)
        gt_code = build_gt_instance_code(
            cur_cls_name,
            gt_event,
            ex['entity_mentions'],
            event_ontology,
            instantiation_prompt,
            args=args
        )

        # 3.3 Build data fields
        data = {
            "line_idx": line_idx,
            "event_idx": event_idx,
            "doc_id": doc_id,
            "sent_id": sent_id,
            "parent_cls_name": event_ontology["parent_cls_name"],
            "cur_cls_name": cur_cls_name,
            "event_mentions": ex["event_mentions"],  # gt data
            "entity_mentions": ex["entity_mentions"],  # gt data
            "gt_instance_code": gt_code,

            "sentence": sentence,
            "code_context": code_context,
            "text_prompt": text_prompt,
            "instantiation_prompt": instantiation_prompt,
            "full_prompt": full_prompt,
        }

        # 3.4 Save data and code to file
        fwrite.write(json.dumps(data) + "\n")
        with open(os.path.join(output_code_dir, f"{line_idx}_{event_idx}.py"), "w") as f:
            f.write(f"# Doc ID: {doc_id} \n# Sent ID: {sent_id}\n")
            f.write(full_prompt)
        with open(os.path.join(output_code_dir, f"{line_idx}_{event_idx}_gt.py"), "w") as f:
            f.write(f"# Doc ID: {doc_id} \n# Sent ID: {sent_id}\n")
            f.write(code_context + text_prompt)
            f.write(gt_code)
        with open(os.path.join(output_code_dir, f"{line_idx}_{event_idx}.json"), "w") as f:
            f.write(json.dumps(data, indent=4) + "\n")


def process_sentence_events(
    ex,
    code_context: str,
    name_to_ontology: Mapping[str, dict],
    line_idx: int,
    output_code_dir: str,
    fwrite,
    args
):
    # skip examples without event mentions
    if len(ex['event_mentions']) < 1:
        return

    sentence = ex["sentence"]
    doc_id = ex["doc_id"]
    sent_id = ex["sent_id"]

    # 1. Build groundtruth code for all event mentions in the sentence (for in-context learning)
    gt_code: str = ""
    cur_cls_names: List[str] = []
    parent_cls_names: List[str] = []
    for i in range(len(ex['event_mentions'])):
        evt_type = ex['event_mentions'][i]['event_type']
        if evt_type not in name_to_ontology:  # skip rare event type
            print(f"Rare event type: {evt_type}")
            continue

        event = name_to_ontology[evt_type]
        cur_cls_name = event["cur_cls_name"]
        cur_cls_names.append(cur_cls_name)
        parent_cls_name = event["parent_cls_name"]
        parent_cls_names.append(parent_cls_name)

        gt_code += build_gt_instance_code(
            cur_cls_name,
            ex['event_mentions'][i],
            ex['entity_mentions'],
            define_event_var=False
        ).rstrip() + ",\n"

    gt_code = "\n".join([
        " " * 4 + line  # indent each line of code to fit in a list
        for line in gt_code.split("\n") if line.strip() != ""
    ])
    gt_code = f"events: List[Event] = [\n{gt_code}\n]\n"

    # 2. Build code prompt from sentence
    # We don't need cur_cls_name (i.e. don't need to know what event type it is)
    text_prompt, instantiation_prompt = build_prompt(
        cur_cls_name=None,
        sentence=sentence,
        args=args
    )
    full_prompt = code_context + text_prompt + instantiation_prompt

    # 3.3 Build data fields
    data = {
        "line_idx": line_idx,
        "event_idx": -1,  # -1 means this is a sentence-level example
        "doc_id": doc_id,
        "sent_id": sent_id,

        "parent_cls_name": parent_cls_names,
        "cur_cls_name": cur_cls_names,

        "event_mentions": ex["event_mentions"],  # gt data
        "entity_mentions": ex["entity_mentions"],  # gt data
        "gt_instance_code": gt_code,

        "sentence": sentence,
        "code_context": code_context,
        "text_prompt": text_prompt,
        "instantiation_prompt": instantiation_prompt,
        "full_prompt": full_prompt,
    }

    # 3.4 Save data and code to file
    fwrite.write(json.dumps(data) + "\n")
    with open(os.path.join(output_code_dir, f"{line_idx}.py"), "w") as f:
        f.write(f"# Doc ID: {doc_id} \n# Sent ID: {sent_id}\n")
        f.write(full_prompt)
    with open(os.path.join(output_code_dir, f"{line_idx}_gt.py"), "w") as f:
        f.write(f"# Doc ID: {doc_id} \n# Sent ID: {sent_id}\n")
        f.write(code_context + text_prompt)
        f.write(gt_code)
    with open(os.path.join(output_code_dir, f"{line_idx}.json"), "w") as f:
        f.write(json.dumps(data, indent=4) + "\n")


def main(args):
    # 1. Load ontology
    with open(args.parsed_ace_roles) as f:
        parsed_ace_roles = json.load(f)
    name_to_ontology = {parsed["name"]: parsed for parsed in parsed_ace_roles}
    print("Ontology loaded.")

    with open(".".join(args.parsed_ace_roles.split(".")[:-1]) + ".py") as f:
        ace_roles_full_context = f.read()

    # 2. Figureout output filepaths
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

            if args.predict_event_type:
                process_sentence_events(
                    ex,
                    ace_roles_full_context,
                    name_to_ontology,
                    line_idx,
                    output_code_dir,
                    fwrite,
                    args
                )
            else:
                process_single_event(
                    ex,
                    name_to_ontology,
                    line_idx,
                    output_code_dir,
                    fwrite,
                    ace_roles_full_context,
                    args
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parsed-ace-roles', required=True)
    parser.add_argument('--input-filepath', required=True)
    parser.add_argument('--output-filedir', required=True)
    parser.add_argument('--reduce-hallucination', action='store_true')
    parser.add_argument('--predict-event-type', action='store_true')
    parser.add_argument('--pure-text-prompt', action='store_true')
    parser.add_argument('--mark-trigger', action='store_true')
    parser.add_argument('--event-detection', action='store_true')
    parser.add_argument('--informative-entity-type', action='store_true')
    parser.add_argument('--add-amr', action='store_true')
    args = parser.parse_args()

    if args.add_amr:
        AMR_STOG = amrlib.load_stog_model()

    main(args)
