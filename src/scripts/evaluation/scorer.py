"""
Modified from: https://github.com/raspberryice/gen-arg/blob/main/src/genie/scorer.py

Scorer for argument extraction on ACE & KAIROS.
For the RAMS dataset, the official scorer is used. 

Outputs: 
Head F1 
Coref F1
"""
import json
import argparse
import logging
import pandas as pd
import spacy
from collections import defaultdict
from typing import List, Any, Mapping, Set, Tuple
from tqdm import tqdm

from src.utils.eval import (
    find_arg_span,
    compute_f1,
    get_entity,
    find_head,
    clean_span,
    WhitespaceTokenizer
)
from src.utils.gen_parse import (
    EventPredictedArgs,
    process_instance_codes_to_args
)
from src.utils.visualize import visualize_predicted_and_gold_entities

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

logging.getLogger().setLevel(logging.ERROR)


def load_data(args):
    def _get_doc_key(example):
        return example["sent_id"] if "sent_id" in example else example["doc_id"]

    ontology_dict = json.load(open(args.ontology_path))
    ontology_dict = {d["name"]: d for d in ontology_dict}
    # ontology_dict = load_ontology(dataset=args.dataset)

    if args.dataset == 'KAIROS' and args.coref and not args.coref_file:
        print('coreference file needed for the KAIROS dataset.')
        raise ValueError
    if args.dataset == 'AIDA' and args.coref:
        raise NotImplementedError

    examples: Mapping[int, Mapping[str, Any]] = {}
    doc2ex = defaultdict(list)  # a document contains multiple events
    # 1. Load generated data
    with open(args.gen_file, 'r') as f:
        # this solution relies on keeping the exact same order
        for lidx, line in enumerate(f):
            pred = json.loads(line.strip())
            examples[lidx] = pred
            input_example = pred["input"]

            # use doc_key as the key to connect prediction AND groundtruth
            doc_key = _get_doc_key(input_example)
            examples[lidx]["doc_key"] = doc_key
            doc2ex[doc_key].append(lidx)

    # 2. Load test (ground truth) data
    exp_id = args.gen_file.split('/')[-2]
    print(f'Evaluating exp_id: {exp_id}')

    if "pevent" in exp_id:
        args.predict_event = True
        print(f"Keyword 'pevent' detected: Predicting event type")
    else:
        args.predict_event = False

    with open(args.test_file, 'r') as f:
        for line in f:
            doc = json.loads(line.strip())
            doc_key = _get_doc_key(doc)

            if args.predict_event and len(doc2ex[doc_key]) > 0:
                assert len(doc2ex[doc_key]) == 1
                eid = doc2ex[doc_key][0]
                examples[eid]['tokens'] = doc['tokens']
                examples[eid]['event'] = doc['event_mentions']
                assert examples[eid]['input']['event_mentions'] == examples[eid]['event']
                examples[eid]['entity_mentions'] = doc['entity_mentions']
                assert examples[eid]["input"]["entity_mentions"] == examples[eid]['entity_mentions']
            else:
                if len(doc2ex[doc_key]) < 1 and len(doc2ex[doc_key]) != len(doc['event_mentions']):
                    print(
                        f'Warning (exp_id: {exp_id}): No prediction found for {doc_key}')
                    continue
                for eid in doc2ex[doc_key]:
                    event_idx = examples[eid]["input"]["event_idx"]

                    examples[eid]['tokens'] = doc['tokens']
                    examples[eid]['event'] = doc['event_mentions'][event_idx]
                    assert examples[eid]['input']['event_mentions'][event_idx] == examples[eid]['event']

                    _input_event_type = examples[eid]['input']['parent_cls_name'].replace("_", "-") \
                        + ":" + \
                        examples[eid]['input']['cur_cls_name'].replace(
                            "_", "-")
                    assert examples[eid]['event']["event_type"] == _input_event_type, \
                        f"Input event type: {_input_event_type} != {examples[eid]['event']['event_type']}"

                    examples[eid]['entity_mentions'] = doc['entity_mentions']
                    assert examples[eid]["input"]["entity_mentions"] == examples[eid]['entity_mentions']

        if "sent_id" in doc.keys():
            print('Evaluating on sentence level')
        else:
            print('Evaluating on document level')

    # span to canonical entity_id mapping for each doc
    coref_mapping: Mapping[str,
                           Mapping[Tuple[int, int], str]] = defaultdict(dict)
    if args.coref:
        if args.dataset == 'KAIROS' and args.coref_file:
            with open(args.coref_file, 'r') as f, open(args.test_file, 'r') as test_reader:
                for line, test_line in zip(f, test_reader):
                    coref_ex = json.loads(line)
                    ex = json.loads(test_line)
                    doc_id = coref_ex['doc_key']

                    for cluster, name in zip(coref_ex['clusters'], coref_ex['informative_mentions']):
                        canonical = cluster[0]
                        for ent_id in cluster:
                            entity = get_entity(ex, ent_id)
                            ent_span = (entity["start"], entity["end"]-1)
                            coref_mapping[doc_id][ent_span] = canonical
                    # this does not include singleton clusters
        else:
            # for the ACE dataset
            with open(args.test_file) as f:
                for line in f:
                    doc = json.loads(line.strip())
                    doc_id = doc['sent_id']
                    for entity in doc['entity_mentions']:
                        mention_id = entity['id']
                        ent_id = '-'.join(mention_id.split('-')[:-1])
                        # all indexes are inclusive
                        coref_mapping[doc_id][(
                            entity['start'], entity['end']-1)] = ent_id

    return examples, coref_mapping, ontology_dict


def construct_pred_set(predicted_args, cur_event, context_words, doc, args):
    # get trigger
    # extract argument span
    trigger_start = cur_event['trigger']['start']
    trigger_end = cur_event['trigger']['end']
    predicted_set = set()

    lowercased_context_words = [w.lower() for w in context_words]
    lowercased_doc = nlp(' '.join(lowercased_context_words)
                         ) if args.head_only else None

    not_matched_pred_args = []
    for argname in predicted_args:
        # this argument span is inclusive, FIXME: this might be problematic
        for entity_type, entity_text in predicted_args[argname]:
            if entity_text is None:  # means no entity is found for this argument
                continue
            entity_text: List[str]
            arg_span = find_arg_span(
                entity_text,
                context_words,
                trigger_start,
                trigger_end,
                head_only=args.head_only,
                doc=doc
            )

            # Attempt to fixed due to cases or symbols in the text
            # e.g., entity = "anwar" vs context_words = "Anwar"
            # e.g., entity = ["Anne-Marie"] vs context_words = ["Anne", "-", "Marie"]
            # e.g., entity = ["roh", "moo-hyun"] vs context_words = ["roh", "moo", "-", "hyun"]
            if not arg_span:
                normalized_entity_text = []
                for word in entity_text:
                    word = word.lower()
                    # process hyphenated words
                    if "-" in word and len(word) > 1:
                        normalized_entity_text.extend(
                            word.replace("-", " - ").split())
                    else:
                        normalized_entity_text.append(word)
                    # TODO: If we really want higher performance on ACE05,
                    # we could fix U.S. -> U.S, british -> british, etc.
                arg_span = find_arg_span(
                    normalized_entity_text,
                    lowercased_context_words,
                    trigger_start,
                    trigger_end,
                    head_only=args.head_only,
                    doc=lowercased_doc
                )

            if arg_span:  # if None means hullucination
                predicted_set.add(
                    (arg_span[0], arg_span[1],
                     cur_event["event_type"], argname, entity_type)
                )
            else:
                not_matched_pred_args.append({
                    "role": argname,
                    "entity_type": entity_type,
                    "text": entity_text
                })
            # With code generation, we don't need to care for "and"
    return predicted_set, not_matched_pred_args


def construct_gold_set(ex, doc, cur_event, doc_key, args):
    gold_set = set()
    # set of canonical mention ids, singleton mentions will not be here
    gold_canonical_set = set()
    for arg in cur_event['arguments']:
        argname = arg['role']
        entity_id = arg['entity_id']
        entity = get_entity(ex, entity_id)

        span = (entity["start"], entity["end"]-1)  # convert to inclusive span
        # clean up span by removing `a` `the`
        span = clean_span(ex, span)

        if args.head_only and span[0] != span[1]:
            span = find_head(span[0], span[1], doc=doc)

        gold_set.add(
            (span[0], span[1], cur_event["event_type"], argname, entity["entity_type"]))

        if args.coref:
            if span in coref_mapping[doc_key]:
                canonical_id = coref_mapping[doc_key][span]
                gold_canonical_set.add(
                    (canonical_id, cur_event["event_type"], argname, entity["entity_type"]))

    return gold_set, gold_canonical_set


def match_pred_gold_sets(
    predicted_set: Set[Tuple[int, int, str, str]],
    gold_set: Set[Tuple[int, int, str, str]],
    gold_canonical_set: Set[Tuple[str, str, str]],
    context_words: List[str],
    doc_key: str,
    stats: dict,
    coref_mapping: Mapping[str, Mapping[Tuple[int, int], str]],
    args: argparse.Namespace,
):
    predicted_arg_stats = []
    for pred_arg in predicted_set:
        arg_start, arg_end, event_type, role, entity_type = pred_arg

        cur_arg_stat = {
            "argument": pred_arg,
            "text": " ".join(context_words[arg_start:arg_end+1]),
            "correct_identification": False,
            "correct_classification": False,
            "correct_ner": False,
            "correct_cls+ner": False,
        }

        # 1. Check Argument Identification (Span + Event Type)
        gold_idn = {
            item for item in gold_set
            if item[0] == arg_start and item[1] == arg_end  # span matched
            and item[2] == event_type  # event type matched
        }

        if gold_idn:
            # Identification is correct
            cur_arg_stat["correct_identification"] = True
            stats["arg_idn_num"] += 1

            # 2. Check Argument Classification (Span + Event Type + Role)
            gold_class = {
                item
                for item in gold_idn
                if item[3] == role  # role matched
            }
            if gold_class:
                # an gold argument is indentified and assigned to correct role
                stats["arg_class_num"] += 1
                cur_arg_stat["correct_classification"] = True

            # 3. Check Argument Identification + NER (Span + Event Type + Entity Type)
            gold_ner = {
                item
                for item in gold_idn
                if item[4] == entity_type  # entity type matched
            }
            if gold_ner:
                # an gold argument is indentified and assigned to correct role
                stats["arg_ner_num"] += 1
                cur_arg_stat["correct_ner"] = True

            # 4. Check Argument Classification + NER (Span + Event Type + Role + Entity Type)
            gold_cls_ner = {
                item
                for item in gold_class
                if item[4] == entity_type  # entity type matched
            }
            if gold_cls_ner:
                # an gold argument is indentified and assigned to correct role
                stats["arg_class+ner_num"] += 1
                cur_arg_stat["correct_cls+ner"] = True

        elif args.coref:  # check coref matches

            pred_span = (arg_start, arg_end)

            # 1. Check Argument Identification (Span + Event Type)
            if pred_span in coref_mapping[doc_key]:
                canonical_id = coref_mapping[doc_key][pred_span]
                gold_idn_coref = {
                    item
                    for item in gold_canonical_set
                    if item[0] == canonical_id and item[1] == event_type
                }

                if gold_idn_coref:
                    stats["arg_idn_coref_num"] += 1
                    cur_arg_stat["correct_identification"] = True

                    # 2. Check Argument Classification (Span + Event Type + Role)
                    gold_class_coref = {
                        item
                        for item in gold_idn_coref
                        if item[2] == role
                    }
                    if gold_class_coref:
                        stats["arg_class_coref_num"] += 1
                        cur_arg_stat["correct_classification"] = True

                    # 3. Check Argument Identification + NER (Span + Event Type + Entity Type)
                    gold_ner_coref = {
                        item
                        for item in gold_idn_coref
                        if item[3] == entity_type
                    }
                    if gold_ner_coref:
                        stats["arg_ner_coref_num"] += 1
                        cur_arg_stat["correct_ner"] = True

                    # 4. Check Argument Classification + NER (Span + Event Type + Role + Entity Type)
                    gold_cls_ner_coref = {
                        item
                        for item in gold_class_coref
                        if item[3] == entity_type
                    }
                    if gold_cls_ner_coref:
                        stats["arg_class+ner_coref_num"] += 1
                        cur_arg_stat["correct_cls+ner"] = True

        predicted_arg_stats.append(cur_arg_stat)

    predicted_arg_stats = sorted(
        predicted_arg_stats,
        key=lambda x: (x["argument"][0], x["argument"][1])
    )  # sort by span start and end for output consistency

    return predicted_arg_stats


def eval_examples(examples, coref_mapping, ontology_dict, args):
    stats = {
        "pred_arg_num": 0,  # number of predicted arguments
        "gold_arg_num": 0,  # number of gold arguments
        "hallucinate_arg_num": 0,  # number of hallucinated arguments

        "arg_idn_num": 0,  # number of correct argument identification
        # number of correct argument NER (span + event type + role)
        "arg_ner_num": 0,
        "arg_class_num": 0,  # number of correct argument classification
        "arg_class+ner_num": 0,  # number of correct argument classification and NER

        # number of correct argument identification (coref)
        "arg_idn_coref_num": 0,
        # number of correct argument NER (span + event type + role) (coref)
        "arg_ner_coref_num": 0,
        # number of correct argument classification (coref)
        "arg_class_coref_num": 0,
        # number of correct argument classification and NER (coref)
        "arg_class+ner_coref_num": 0,
    }

    errors = []
    warnings = []
    print("\n# Test Instances", file=OUTPUT_FILE)
    for ex_id, ex in tqdm(enumerate(examples.values()), total=len(examples)):
        # 1. Parse and Get predicted arguments
        context_words = ex['tokens']
        doc_key = ex['doc_key']
        if args.head_only:
            doc = nlp(' '.join(context_words))
        else:
            doc = None

        (
            predicted_args,
            role_to_cluster_strings,
            parsing_warnings,
            partial_error
        ) = process_instance_codes_to_args(
            ex, doc_key, warnings, errors, args
        )

        if not args.predict_event:
            cur_event = ex["event"]
            if cur_event["event_type"] not in ontology_dict:
                warnings.append(
                    f"WARNING: {cur_event['event_type']} is not in the ontology")
                continue

            # 2. Construct Predicted Set
            predicted_set, not_matched_pred_args = construct_pred_set(
                predicted_args, cur_event, context_words, doc, args
            )
            stats["pred_arg_num"] += len(predicted_set)
            stats["hallucinate_arg_num"] += len(not_matched_pred_args)

            # 3. Construct Gold Set (ground truth)
            gold_set, gold_canonical_set = construct_gold_set(
                ex, doc, cur_event, doc_key, args)
            stats["gold_arg_num"] += len(gold_set)

            # 4. Check matches between predicted set AND gold set
            _prev_arg_idn_num = stats["arg_idn_num"]
            _prev_arg_class_num = stats["arg_class_num"]
            _prev_arg_clsner_num = stats["arg_class+ner_num"]
            _prev_arg_idn_coref_num = stats["arg_idn_coref_num"]
            _prev_arg_class_coref_num = stats["arg_class_coref_num"]
            _prev_arg_clsner_coref_num = stats["arg_class+ner_coref_num"]

            predicted_arg_stats = match_pred_gold_sets(
                predicted_set, gold_set, gold_canonical_set, context_words, doc_key, stats, coref_mapping, args
            )
            arg_idn_num = stats["arg_idn_num"] - _prev_arg_idn_num
            arg_class_num = stats["arg_class_num"] - _prev_arg_class_num
            arg_clsner_num = stats["arg_class+ner_num"] - _prev_arg_clsner_num
            arg_idn_coref_num = stats["arg_idn_coref_num"] - \
                _prev_arg_idn_coref_num
            arg_class_coref_num = stats["arg_class_coref_num"] - \
                _prev_arg_class_coref_num
            arg_clsner_coref_num = stats["arg_class+ner_coref_num"] - \
                _prev_arg_clsner_coref_num

        else:  # predict_event = True (i.e., multiple events in one predictions)
            assert isinstance(predicted_args, list)
            predicted_args: List[EventPredictedArgs]

            for cur_event in ex["events"]:
                if cur_event["event_type"] not in ontology_dict:
                    warnings.append(
                        f"WARNING: {cur_event['event_type']} is not in the ontology")
                    continue
            # TODO: fix this
            raise NotImplementedError

        # 5. Output relevant eval information for this instance
        correct_identification = all(
            [item["correct_identification"] for item in predicted_arg_stats])
        correct_classification = all([item["correct_classification"]
                                      for item in predicted_arg_stats])
        OUTPUT_JSONL.write(json.dumps({
            "doc_key": doc_key,
            "event_type": cur_event["event_type"],

            "arg_idn_num": arg_idn_num,  # number of correct argument identification
            "arg_class_num": arg_class_num,  # number of correct argument classification
            # number of correct argument classification and NER
            "arg_clsner_num": arg_clsner_num,

            # number of correct argument identification (coref)
            "arg_idn_coref_num": arg_idn_coref_num,
            # number of correct argument classification (coref)
            "arg_class_coref_num": arg_class_coref_num,
            # number of correct argument classification and NER (coref)
            "arg_clsner_coref_num": arg_clsner_coref_num,

            "gold_arg_num": len(gold_set),  # number of gold arguments
            # number of predicted arguments
            "pred_arg_num": len(predicted_set),

            "correct_identification": correct_identification,
            "correct_classification": correct_classification,
            "predicted_arguments": predicted_arg_stats,
            "not_matched_pred_args": not_matched_pred_args,

            "pred_set": list(predicted_set),
            "gold_arguments": list(gold_set),
            "gold_canonical_arguments": list(gold_canonical_set),

            "role_to_cluster_strings": role_to_cluster_strings,
            "input_example": ex,
        }) + "\n")

        # Identification / Classificaion Correctness
        print(
            f"\n## [I:{correct_identification}/C:{correct_classification}] Instance {ex_id} / {len(examples)}", file=OUTPUT_FILE)
        print(
            f"\n**Test File Line Index**: `{ex['input']['line_idx']}`",
            file=OUTPUT_FILE
        )
        print(f"\n**Doc Key**: `{doc_key}`", file=OUTPUT_FILE)

        predicted_html, gold_html = visualize_predicted_and_gold_entities(
            context_words,
            predicted_arg_stats,
            ex
        )
        print(f"\n**Event Type**: {cur_event['event_type']}", file=OUTPUT_FILE)
        print(f"\n### **Predicted** Arguments\n", file=OUTPUT_FILE)
        print(predicted_html, file=OUTPUT_FILE)
        print(f"\n### **Groundtruth** Arguments\n", file=OUTPUT_FILE)
        print(gold_html, file=OUTPUT_FILE)
        if len(not_matched_pred_args) > 0:
            print(
                f"\n### Not Matched Predicted Arguments (Hallucinations)\n", file=OUTPUT_FILE)
            pd.DataFrame(not_matched_pred_args).to_html(
                OUTPUT_FILE, index=False)
        # Visualize generated codes
        print(
            f"\n### Generated Instance Code\n",
            file=OUTPUT_FILE
        )
        if partial_error:
            print(
                f"\nNOTE: Parsing error detected in some of the generations.\n", file=OUTPUT_FILE)
        for i, _code in enumerate(ex['instance_code']):
            print(f"**Code generation {i+1}**", file=OUTPUT_FILE)
            print(f"```python\n{_code}\n```", file=OUTPUT_FILE)
            if len(parsing_warnings[i]) > 0:
                print(f"\nGeneration Warning:\n", file=OUTPUT_FILE)
                for i, _warnings in enumerate(parsing_warnings):
                    for _warning in _warnings:
                        print(f"> {_warning} \n", file=OUTPUT_FILE)

        print(f"\n### Groundtruth Event\n", file=OUTPUT_FILE)
        print(
            f"```json\n{json.dumps(ex['event'], indent=2)}\n```", file=OUTPUT_FILE)

        if role_to_cluster_strings is not None:
            print(
                f"\n### Clustered Equivalent Strings (from multiple generations)\n"
                f"```json\n{json.dumps(role_to_cluster_strings, indent=4)}\n```",
                file=OUTPUT_FILE
            )

    return (
        stats,
        errors,
        warnings
    )


def performace_to_table(
    role_id_prec, role_id_rec, role_id_f,
    role_prec, role_rec, role_f,
    role_ner_prec, role_ner_rec, role_ner_f,
    role_cls_ner_prec, role_cls_ner_rec, role_cls_ner_f
):
    return pd.DataFrame(
        {
            "Role Identification": [
                role_id_prec * 100.0,
                role_id_rec * 100.0,
                role_id_f * 100.0
            ],
            "Role Classification": [
                role_prec * 100.0,
                role_rec * 100.0,
                role_f * 100.0
            ],
            "Role Identification + NER": [
                role_ner_prec * 100.0,
                role_ner_rec * 100.0,
                role_ner_f * 100.0
            ],
            "Role Classification + NER": [
                role_cls_ner_prec * 100.0,
                role_cls_ner_rec * 100.0,
                role_cls_ner_f * 100.0
            ],
        },
        index=["Precision", "Recall", "F1"]
    )


def calculate_stats(
    stats,
    errors,
    warnings,
    args
):
    print('\n# Parsing Syntax Errors ', file=OUTPUT_FILE)
    print(
        f"Number of Warning Instances: {len(warnings)} / {len(examples.values())} \n", file=OUTPUT_FILE)
    print(
        f"Number of Warning Instances: {len(warnings)} / {len(examples.values())}")
    print(
        f"Number of Error Instances: {len(errors)} / {len(examples.values())} \n", file=OUTPUT_FILE)
    print(
        f"Number of Error Instances: {len(errors)} / {len(examples.values())}")
    print(
        f"Number of Error Instances (finish reason = length): "
        f"{sum(e['output']['choices'][0].get('finish_reason', None) == 'length' for e in errors)} / {len(errors)} \n",
        file=OUTPUT_FILE
    )
    for i, error_example in enumerate(errors):
        print(f"\n## Error {i}\n", file=OUTPUT_FILE)
        print(
            f"Finish Reason: `{json.dumps(error_example['output']['choices'][0].get('finish_reason', None), indent=4)}` \n", file=OUTPUT_FILE)
        print(
            f"\n**Text**\n> {' '.join(error_example['tokens'])} \n", file=OUTPUT_FILE)
        print(
            f"\n**Instance Code**\n",
            file=OUTPUT_FILE
        )
        for i, _code in enumerate(error_example['instance_code']):
            print(f"```python\n{_code}\n```", file=OUTPUT_FILE)
        print(
            f"\n**Groundtruth Event**\n"
            f"```json\n{json.dumps(error_example['event'], indent=4)}\n```",
            file=OUTPUT_FILE
        )

    print("\n\n# Evaluation Result ", file=OUTPUT_FILE)
    if args.head_only:
        print(
            'NOTE: below stats are calculate by matching head words only.',
            file=OUTPUT_FILE
        )

    print(f"\n## Prediction Stats", file=OUTPUT_FILE)
    num_table = pd.DataFrame(
        [
            len(examples.values()),
            len(errors),
            len(warnings),
            stats["gold_arg_num"],
            stats["pred_arg_num"],
            stats["hallucinate_arg_num"],

            stats["arg_idn_num"],
            stats["arg_ner_num"],
            stats["arg_class_num"],
            stats["arg_class+ner_num"],

            # coref
            stats["arg_idn_coref_num"],
            stats["arg_ner_coref_num"],
            stats["arg_class_coref_num"],
            stats["arg_class+ner_coref_num"]
        ],
        index=[
            # Example-level
            "# of Examples",
            "# of Error Examples",
            "# of Warning Examples",

            # Argument-level
            "# of Gold Arguments",
            "# of Predicted Arguments",
            "# of Hallucinated Arguments",
            "# of Correctly Identified Arguments",
            "# of Correctly Identified Arguments + NER",
            "# of Correctly Identified and Classified Arguments",
            "# of Correctly Identified, Classified and NER Arguments",

            "# of Correctly (coref) Identified Arguments",
            "# of Correctly (coref) Identified Arguments + NER",
            "# of Correctly (coref) Identified and Classified Arguments",
            "# of Correctly (coref) Identified, Classified and NER Arguments",
        ],
    )
    print(num_table.to_markdown(), file=OUTPUT_FILE)

    print(f"\n## Performance Stats", file=OUTPUT_FILE)
    if args.coref:
        role_id_prec, role_id_rec, role_id_f = compute_f1(
            stats["pred_arg_num"], stats["gold_arg_num"], stats["arg_idn_num"] +
            stats["arg_idn_coref_num"]
        )
        role_prec, role_rec, role_f = compute_f1(
            stats["pred_arg_num"], stats["gold_arg_num"], stats["arg_class_num"] +
            stats["arg_class_coref_num"]
        )
        role_ner_prec, role_ner_rec, role_ner_f = compute_f1(
            stats["pred_arg_num"], stats["gold_arg_num"], stats["arg_ner_num"] +
            stats["arg_ner_coref_num"]
        )

        role_cls_ner_prec, role_cls_ner_rec, role_cls_ner_f = compute_f1(
            stats["pred_arg_num"], stats["gold_arg_num"], stats["arg_class+ner_num"] +
            stats["arg_class+ner_coref_num"]
        )

        performance_table = performace_to_table(
            role_id_prec, role_id_rec, role_id_f,
            role_prec, role_rec, role_f,
            role_ner_prec, role_ner_rec, role_ner_f,
            role_cls_ner_prec, role_cls_ner_rec, role_cls_ner_f
        )
        print(performance_table.to_markdown(), file=OUTPUT_FILE)
    else:
        # Arguments Identification
        role_id_prec, role_id_rec, role_id_f = compute_f1(
            stats["pred_arg_num"], stats["gold_arg_num"], stats["arg_idn_num"]
        )

        # Arguments Classification
        role_prec, role_rec, role_f = compute_f1(
            stats["pred_arg_num"], stats["gold_arg_num"], stats["arg_class_num"]
        )

        # Arguments Identifiction + NER
        role_ner_prec, role_ner_rec, role_ner_f = compute_f1(
            stats["pred_arg_num"], stats["gold_arg_num"], stats["arg_ner_num"]
        )

        # Arguments Classification + NER
        role_cls_ner_prec, role_cls_ner_rec, role_cls_ner_f = compute_f1(
            stats["pred_arg_num"], stats["gold_arg_num"], stats["arg_class+ner_num"]
        )
        performance_table = performace_to_table(
            role_id_prec, role_id_rec, role_id_f,
            role_prec, role_rec, role_f,
            role_ner_prec, role_ner_rec, role_ner_f,
            role_cls_ner_prec, role_cls_ner_rec, role_cls_ner_f
        )
        print(performance_table.to_markdown(), file=OUTPUT_FILE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gen-file',
        type=str,
        default='checkpoints/gen-all-ACE-freq-pred/predictions.jsonl'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        default='data/ace/zs-freq-10/test.oneie.json'
    )
    parser.add_argument('--ontology-path')  # parsed ace_roles.json
    parser.add_argument('--coref-file', type=str)
    parser.add_argument('--head-only', action='store_true')
    parser.add_argument('--coref', action='store_true')
    parser.add_argument(
        '--dataset',
        type=str,
        default='ACE',
        choices=['ACE', 'KAIROS', 'AIDA']
    )
    args = parser.parse_args()

    gen_name = args.gen_file.split('/')[-2]
    print(f"Evaluating on: {gen_name}")
    args.pure_text = bool("puretext" in gen_name)

    if args.coref:
        assert args.head_only
        _output_filename = args.gen_file + ".coref.eval.md"
        _output_jsonl_path = args.gen_file + ".coref.eval.json"
    elif args.head_only:
        _output_filename = args.gen_file + ".head_only.eval.md"
        _output_jsonl_path = args.gen_file + ".head_only.eval.json"
    else:
        _output_filename = args.gen_file + ".eval.md"
        _output_jsonl_path = args.gen_file + ".eval.json"
    OUTPUT_FILE = open(_output_filename, "w")
    OUTPUT_JSONL = open(_output_jsonl_path, "w")

    print(
        "# Evaluation Arguments:\n"
        f"```json\n{json.dumps(vars(args), indent=4)}\n```",
        file=OUTPUT_FILE
    )

    examples, coref_mapping, ontology_dict = load_data(args)

    (
        stats,
        errors,
        warnings
    ) = eval_examples(examples, coref_mapping, ontology_dict, args)

    calculate_stats(
        stats,
        errors,
        warnings,
        args
    )

    OUTPUT_FILE.close()
    OUTPUT_JSONL.close()
