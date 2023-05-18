import argparse
import logging
import re
import numpy as np
import ast
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Mapping, Tuple
from collections import Counter
from thefuzz import fuzz


class EventPredictedArgs(defaultdict):
    def __init__(self, event_type: str = None):
        super().__init__(list)
        # e.g., Looks like this: {
        # 'Agent': [("PER", ['British', 'Chancellor', 'of', 'the', 'Exchequer', 'Gordon', 'Brown'])]
        # 'Person': [("PER", ['head', 'of', 'the', 'country', "'s", 'energy', 'regulator'])]
        # }
        self.event_type = event_type


def split_text_w_ellipsis_check(text, warnings):
    if text == Ellipsis:  # check whether the text is Ellipsis (e.g., ...)
        warnings.append(
            f"Invalid base entity instance: the text is Ellipsis."
        )
        return None
    if text is not None:
        return str(text).strip().split()
    return None


def _extract_base_entity_instance(base_entity_instance):
    warnings = []

    # check base entity type
    if isinstance(base_entity_instance, ast.Constant) and base_entity_instance.value is None:
        return None, None, warnings

    # 1. Correctly initialize the base entity instance
    if isinstance(base_entity_instance, ast.Call):
        if len(base_entity_instance.keywords) < 1:
            if len(base_entity_instance.args) != 1:  # check
                raise SyntaxError(
                    f"Base entity instance has unsupported {len(base_entity_instance.args)} arguments when no keywords are presence!"
                )

            if not (len(base_entity_instance.args) == 1 and isinstance(
                    base_entity_instance.args[0], ast.Constant)):  # check
                # e.g., Personnel(PER("Milosevic"))
                raise SyntaxError(
                    f"Invalid base entity instance (potentially caused by nesting ast.Call): {base_entity_instance}"
                )

            # e.g., ORG("British Chancellor of the Exchequer Gordon Brown")
            text = base_entity_instance.args[0].value

        else:
            if len(base_entity_instance.keywords) != 1:  # check
                warnings.append(
                    f"Invalid base entity instance: {base_entity_instance} has {len(base_entity_instance.keywords)} keywords."
                )
                return None, None, warnings

            assert isinstance(
                base_entity_instance.keywords[0], ast.keyword)  # check

            if base_entity_instance.keywords[0].arg != 'name':  # check
                warnings.append(
                    f"Expect 'name' as base_entity_instance.keywords[0].arg, instead we get {base_entity_instance.keywords[0].arg}"
                )
                return None, None, warnings

            _kwvalue = base_entity_instance.keywords[0].value
            if isinstance(_kwvalue, ast.Constant):
                # e.g., ORG(name="British Chancellor of the Exchequer Gordon Brown")
                text = _kwvalue.value
            elif isinstance(_kwvalue, ast.List) and len(_kwvalue.elts) == 1:
                # e.g., ORG(name=["British Chancellor of the Exchequer Gordon Brown"])
                text = _kwvalue.elts[0].value
            else:
                warnings.append(
                    f"Invalid base entity instance: a kwvalue with type {_kwvalue.__class__.__name__} doesn't satisfy constraints."
                )
                return None, None, warnings

        try:
            base_entity_type = base_entity_instance.func.id
        except AttributeError:
            warnings.append(
                f"Invalid base entity instance: {base_entity_instance} doesn't have a valid base_entity_type."
            )
            base_entity_type = None

        text = split_text_w_ellipsis_check(text, warnings)
        return base_entity_type, text, warnings

    # 2. Rare case, model directly put the string as the base entity type
    if not isinstance(base_entity_instance, ast.Constant):
        raise SyntaxError(
            f"Invalid base entity instance (potentially caused by nesting ast.Call or ast.List): {base_entity_instance}"
        )
    # e.g., "British Chancellor of the Exchequer Gordon Brown"
    text = split_text_w_ellipsis_check(base_entity_instance.value, warnings)
    base_entity_type = None
    warnings.append(f"WARNING: base entity type is a string: {text}")
    return base_entity_type, text, warnings


def _instance_code_filter_heuristic(instance_code):
    """Attempt to fix some common errors in instance codes."""
    instance_code = instance_code.replace("```", "")
    instance_code = instance_code.replace("[[", "[")
    instance_code = instance_code.replace("]]", "]")

    # Filter lines (only keep lines for instantiation)
    lines = instance_code.split("\n")
    useful_lines = []
    _instantiation_done = False
    for line in lines:
        if not _instantiation_done:
            useful_lines.append(line)
        if line.startswith(")"):
            _instantiation_done = True
    instance_code = "\n".join(useful_lines)

    return instance_code


def _parse_event_instance(
    event_instance: ast.Call,
    parsing_warnings: List[str]
):
    """Parse an event instance.

    Example of an event instance:
    ```
    EventType(
        key1=[EntityType("value1")],
        key2=[EntityType("value2")]
    )
    """
    event_type = event_instance.func.id
    attribute_to_entity_type_text_pairs = defaultdict(list)
    for kwarg in event_instance.keywords:
        # e.g., agent=[ORG(name="British Chancellor of the Exchequer Gordon Brown")]
        attribute = kwarg.arg  # e.g., agent

        # get base instance
        assert isinstance(kwarg, ast.keyword)
        if isinstance(kwarg.value, ast.List):
            base_entity_instance_list = kwarg.value
            # e.g., =[ORG(name="British Chancellor of the Exchequer Gordon Brown")]
            # OR =[ORG("British Chancellor of the Exchequer Gordon Brown")]
            for base_entity_instance in base_entity_instance_list.elts:
                base_entity_type, splited_text, warnings = _extract_base_entity_instance(
                    base_entity_instance)
                parsing_warnings.extend(warnings)
                # update mapping
                if splited_text is not None:
                    attribute_to_entity_type_text_pairs[attribute].append(
                        (base_entity_type, splited_text))

        elif isinstance(kwarg.value, ast.Call):
            # e.g., =ORG(name="British Chancellor of the Exchequer Gordon Brown")
            base_entity_instance = kwarg.value
            base_entity_type, splited_text, warnings = _extract_base_entity_instance(
                base_entity_instance)
            parsing_warnings.extend(warnings)
            parsing_warnings.append(
                f"WARNING: {kwarg.value} is not a list.")
            # update mapping
            if splited_text is not None:
                attribute_to_entity_type_text_pairs[attribute].append(
                    (base_entity_type, splited_text))

        else:
            parsing_warnings.append(
                f"WARNING: {kwarg.value} ({type(kwarg.value)}) is neither a art.Call nor ast.List.")

    # 3. Map attribute to event roles AND build predicted_args
    predicted_args = EventPredictedArgs(event_type=event_type)
    for attr in attribute_to_entity_type_text_pairs.keys():
        role = attr[0].upper() + attr[1:]
        predicted_args[role].extend(attribute_to_entity_type_text_pairs[attr])
    return predicted_args


def parse_instance_code(
    instance_code: str,
    parse_multiple_events: bool = False
) -> Tuple[EventPredictedArgs | List[EventPredictedArgs], List[str]]:
    parsing_warnings = []

    # 1. Parse attributes from instance code
    instance_code = _instance_code_filter_heuristic(instance_code)
    tree = ast.parse(instance_code)

    # 2. Parse event instance
    if len(tree.body) >= 1 and isinstance(tree.body[0], (ast.Assign, ast.AnnAssign)):
        # assert len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign)
        # e.g., nomiate_event = Nominate(...)
        # OR when parse_multiple_events=True
        # e.g., events: List[Event] = [Nominate(...), Nominate(...)]
        _assignment = tree.body[0]
    else:
        raise SyntaxError(
            f"Invalid instance code as its first element is NOT `ast.Assign`: {instance_code}"
        )

    if not parse_multiple_events:
        # expect _assignment.value to be ast.Call
        if isinstance(_assignment.value, ast.Assign):
            parsing_warnings.append(
                f"WARNING: instance code has nested ast.Assign - go down one level: {instance_code}"
            )
            _assignment = _assignment.value

        if isinstance(_assignment.value, ast.Tuple):  # handle weird cases
            # e.g., elect_event = Elect(entity=[ORG("government")],),
            parsing_warnings.append(
                f"WARNING: assignment.value has ast.Tuple, taking its first element: {instance_code}"
            )
            _event_instance = _assignment.value.elts[0]
        else:
            _event_instance = _assignment.value

        # now _event_instance is ast.Call. e.g., Elect(entity=[ORG("government")])
        if not isinstance(_event_instance, ast.Call):
            raise SyntaxError(
                f"Invalid instance code as assignment.value is NOT ast.Call, but {type(_event_instance)}): {instance_code}"
            )
        predicted_args = _parse_event_instance(
            _event_instance, parsing_warnings)

    else:  # parse_multiple_events=True
        # expect _assignment.value to be ast.List
        if not isinstance(_assignment.value, ast.List):
            raise SyntaxError(
                f"Invalid instance code as assignment.value is NOT ast.List, but {type(_assignment.value)}): {instance_code}"
            )
        _event_instance_list = _assignment.value
        # [Nominate(...), Nominate(...)]
        predicted_args: List[EventPredictedArgs] = []
        for _event_instance in _event_instance_list.elts:
            predicted_args.append(
                _parse_event_instance(_event_instance, parsing_warnings)
            )

    return predicted_args, parsing_warnings


def parse_pure_text(text: str) -> Tuple[EventPredictedArgs, List[str]]:
    """Parse pure text into a mapping from event roles to text.

    Args:
        text: a string of text.

    Returns:
        A mapping from event roles to text.
    """
    def _text_post_process(text):
        text = text.replace("'s", f" 's")
        return text

    predicted_args = EventPredictedArgs()
    warnings = []
    for line in text.split("\n"):
        # e.g., line
        # 1. agent: (GPE) "British", (ORG) "the Financial Services Authority (FSA)"
        try:
            attribute, predictions = line.split(":")
            attribute = attribute.split(".")[-1].strip()  # e.g., agent
            # Need to capitalize the first letter (for consistency)
            role = attribute[0].upper() + attribute[1:]  # e.g., Agent
        except Exception as e:
            warnings.append(f"WARNING: failed to parse line: {line}")
            continue

        # (GPE) "British", (ORG) "the Financial Services Authority (FSA)"
        # use regex to extract (GPE) "British" and (ORG) "the Financial Services Authority (FSA)"
        predictions = predictions.strip()
        for match in re.finditer(r"\((.*?)\)\s*\"(.*?)\"", predictions):
            entity_type, entity_text = match.groups()
            entity_text = _text_post_process(entity_text)
            if entity_text != "":
                # e.g., (GPE, ["British"])
                predicted_args[role].append((entity_type, entity_text.split()))
    return predicted_args, warnings


def _reduce_cluster(cluster: Tuple[int], type_str_pairs: List[str]) -> str:
    """"""
    # e.g., cluster = (0, 1)
    assert len(cluster) > 0

    # e.g., type_str_pairs = [(GPE, "British"), (ORG, "the Financial Services Authority (FSA)")]
    # Select the most frequent predicted type
    most_common_type = Counter([
        type_str_pairs[i][0] for i in cluster
    ]).most_common(1)[0][0]  # most_common(1) returns a list of [(type, count)]

    # Select the longest string with the most common type in the cluster
    selected_cluster_str = max([
        type_str_pairs[idx][1] for idx in cluster
    ], key=len)

    # NOTE: we have to split back to list of tokens for later processing
    return (most_common_type, selected_cluster_str.split())


def filter_args_by_frequency(
    args_list: List[EventPredictedArgs],
    similarity_threshold: float = 0.7,
) -> Tuple[EventPredictedArgs, List[str]]:
    """Filter out arguments that didn't occur too much on multiple generations."""
    n_generations = len(args_list)

    # 1. Unravel all arguments (from multiple generations) as list of string
    role_to_type_str_pairs = defaultdict(list)
    _prev_event_type = None
    for generation_id, predicted_args in enumerate(args_list):
        predicted_args: EventPredictedArgs
        if predicted_args is None:
            # skip when parsing failed for this generation
            continue

        # check event type consistency
        if _prev_event_type is not None:
            assert _prev_event_type == predicted_args.event_type
        _prev_event_type = predicted_args.event_type

        # e.g., predicted_args = {
        # 'Agent': [("PER", ['British', 'Chancellor', 'of', 'the', 'Exchequer', 'Gordon', 'Brown'])]
        # 'Person': [("PER", ['head', 'of', 'the', 'country', "'s", 'energy', 'regulator'])]
        # }
        for role, type_tokens_pairs in predicted_args.items():
            for entity_type, tokens in type_tokens_pairs:
                role_to_type_str_pairs[role].append(
                    (entity_type, " ".join(tokens)))

    role_to_cluster_type_str_pairs = {}
    role_to_output = EventPredictedArgs(event_type=_prev_event_type)
    for role, type_str_pairs in role_to_type_str_pairs.items():
        # e.g., type_str_pairs = [
        # ("PER", 'British Chancellor of the Exchequer Gordon Brown'),
        # ("PER', Gordon Brown')
        # ]
        n = len(type_str_pairs)
        # 2. Calculate similarity between all pairs of strings via fuzzy matching (levenshtein distance)
        # count_vec = CountVectorizer(stop_words="english").fit_transform(list_of_str).toarray()
        # similarity_mat = cosine_similarity(count_vec)
        similarity_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # https://towardsdatascience.com/string-matching-with-fuzzywuzzy-e982c61f8a84
                similarity_mat[i, j] = similarity_mat[j, i] = fuzz.ratio(
                    # similarity between strings
                    type_str_pairs[i][1], type_str_pairs[j][1]
                ) / 100.0
        assert similarity_mat.shape == (n, n)

        # 3. Build unique clusters where each cluster is a set of indices that are similar to each other
        equivalent_clusters = set()
        for arg_id in range(n):
            # find all similar arguments
            similar_args = np.where(
                similarity_mat[arg_id] > similarity_threshold
            )[0]
            # add to equivalent clusters
            equivalent_clusters.add(tuple(sorted(similar_args)))

        role_to_cluster_type_str_pairs[role] = [
            [type_str_pairs[str_idx] for str_idx in cluster]
            for cluster in equivalent_clusters
        ]

        # 4. Find cluster with occurences that are more than (total number of generations) // 2
        # e.g., if there are 3 generations, then we want to keep the arguments that occur at least 2 times
        filtered_equivalent_clusters = [
            cluster for cluster in equivalent_clusters
            if len(cluster) > (n_generations // 2)
        ]

        # 5. Reduce Cluster
        # Within cluster, break ties by selecting the longest string & most frequent entity type
        role_to_output[role] = [
            _reduce_cluster(cluster, type_str_pairs)
            for cluster in filtered_equivalent_clusters
        ]
    return role_to_output, role_to_cluster_type_str_pairs


def process_instance_codes_to_args(
    ex, doc_key, warnings: list, errors: list, args: argparse.Namespace
):
    n_errors = 0
    parsing_warnings = []
    predicted_args_prefilter: List[EventPredictedArgs] = []
    partial_error = False
    for cur_instance_code in ex['instance_code']:
        try:
            if args.pure_text:
                cur_predicted_args, cur_warnings = parse_pure_text(
                    cur_instance_code
                )
            else:
                cur_predicted_args, cur_warnings = parse_instance_code(
                    cur_instance_code,
                    parse_multiple_events=args.predict_event
                )
            cur_predicted_args: EventPredictedArgs
            predicted_args_prefilter.append(cur_predicted_args)
            parsing_warnings.append(cur_warnings)
        except SyntaxError as e:
            logging.warning(
                f"Syntax error in instance code (doc_key {doc_key})"
            )
            n_errors += 1
            parsing_warnings.append([])
            continue
    if any(len(i) > 0 for i in parsing_warnings):
        logging.warning(
            f"Parsing warning detected in instance code (doc_key {doc_key})")
        warnings.append(ex)
    if n_errors > 0:
        # all instance codes are invalid
        if n_errors == len(ex['instance_code']):
            logging.warning(
                f"All instance codes generated are invalid w/ syntax error (doc_key {doc_key})")
            errors.append(ex)
        else:
            logging.warning(
                f"Some instance codes generated are invalid w/ syntax error (doc_key {doc_key})")
            partial_error = True

    if args.predict_event:
        if len(predicted_args_prefilter) > 1:
            # There's multiple generations of multiple events (e.g., 3 generations, each 2 events)
            raise NotImplementedError(
                "Evaluating multiple generations of multiple events are not supported yet"
            )
        # TODO: don't apply post-filterings, since it could be hard to interpret
        predicted_args: List[EventPredictedArgs] = predicted_args_prefilter[0]
        role_to_cluster_strings = {}
    else:
        predicted_args: EventPredictedArgs
        predicted_args, role_to_cluster_strings = filter_args_by_frequency(
            predicted_args_prefilter
        )
    # predicted_args: Dict[str, List[List[str]]] -> dict map to [(entity_type, list of tokens), ...]
    return predicted_args, role_to_cluster_strings, parsing_warnings, partial_error


def process_instance_code_to_trigger(
    ex: dict, doc_key, warnings: list, errors: list, args: argparse.Namespace
):
    if len(ex['instance_code']) != 1:
        raise NotImplementedError(
            f"Multiple instance codes generated for trigger (doc_key {doc_key})")

    instance_code = ex['instance_code'][0].rstrip() + "\n)"
    # e.g. instance code
    # assert_event_trigger_words_and_type(
    #     "British Chancellor of the Exchequer Gordon Brown on Tuesday named the current head of the country 's energy regulator as the new chairman of finance watchdog the Financial Services Authority ( FSA ) .",
    #     ["named"],
    #     Elect
    # )

    try:
        parsed = ast.parse(instance_code)
        # parse the 2nd and 3rd arguments
        call: ast.Call = parsed.body[0].value
        if not isinstance(call, ast.Call):
            errors.append(ex)
            return None
        ast_text, ast_trigger_words, ast_event_type = call.args
        assert isinstance(
            ast_text, ast.Constant) and ast_text.value == ex["input"]["sentence"]
        assert isinstance(ast_trigger_words, ast.List)
        assert isinstance(ast_event_type, ast.Name)
        trigger_words = [i.value for i in ast_trigger_words.elts]
        event_type = ast_event_type.id

        return {
            "trigger_words": trigger_words,
            "event_type": event_type
        }

    except SyntaxError as e:
        logging.warning(
            f"Syntax error in instance code (doc_key {doc_key})"
        )
        n_errors += 1
    return None
