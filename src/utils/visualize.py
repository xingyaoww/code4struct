import seaborn as sns
from spacy import displacy

from typing import List, Dict, Tuple, Mapping

from src.utils.eval import get_entity


def get_predicted_char_spans(context_words: List[str], predicted_args):
    """
    Example input:

    context_words = ["In", "a", "tearful", ...]
    predicted_args = [
        {
            "argument": [
                8,
                8,
                "Movement:Transport",
                "Destination"
            ],
            "text": "Vancouver",
            "correct_identification": True,
            "correct_classification": True
        },
        ...
    ]
    """
    predicted_char_spans = []
    for arg in predicted_args:
        start = sum(
            len(word) + 1 for word in context_words[:arg["argument"][0]])
        end = start + len(arg["text"])
        predicted_char_spans.append({
            "start": start,
            "end": end,
            # entity type + role type
            "label": str(arg["argument"][4]) + " " + arg["argument"][3],
        })
    return predicted_char_spans


def word_start_to_char_start(context_words, word_start):
    return sum(len(word) + 1 for word in context_words[:word_start])


def get_gold_char_spans(context_words: List[str], ex):
    """
    Example input:

    context_words = ["In", "a", "tearful", ...]
    gold_args = [
        {
            "entity_id": "APW_ENG_20030324.0768-E6-15",
            "text": "Vancouver",
            "role": "Destination"
        },
        ...
    ]
    """
    gold_char_spans = []
    for arg in ex["event"]["arguments"]:
        entity_mention = get_entity(ex, arg["entity_id"])
        word_start = entity_mention["start"]
        start = word_start_to_char_start(context_words, word_start)
        end = start + len(arg["text"])

        gold_char_spans.append({
            "start": start,
            "end": end,
            # role type
            "label": entity_mention["entity_type"] + " " + arg["role"],
        })
    return gold_char_spans


def build_color_mapping_for_roles(unique_roles: List[str]) -> Mapping[str, str]:
    color_palette = sns.color_palette().as_hex()

    role_colormap = {}
    for i, role in enumerate(unique_roles):
        role_colormap[role] = color_palette[i % len(color_palette)]
    return role_colormap


def visualize_predicted_and_gold_entities(
    context_words,
    predicted_args,
    ex
):
    predicted_char_spans = get_predicted_char_spans(
        context_words, predicted_args)
    gold_char_spans = get_gold_char_spans(context_words, ex)
    unique_roles = list(
        set([span["label"] for span in predicted_char_spans + gold_char_spans]))

    role_colormap = build_color_mapping_for_roles(unique_roles)
    predicted_html = displacy.render(
        {"text": " ".join(context_words), "ents": predicted_char_spans},
        style="ent",
        manual=True,
        minify=True,
        jupyter=False,
        options={
            "colors": role_colormap,
            "ents": unique_roles,
        },
    )

    gold_html = displacy.render(
        {"text": " ".join(context_words), "ents": gold_char_spans},
        style="ent",
        manual=True,
        minify=True,
        jupyter=False,
        options={
            "colors": role_colormap,
            "ents": unique_roles,
        },
    )

    return predicted_html, gold_html
