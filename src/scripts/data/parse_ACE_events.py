"""
Convert ACE event (with roles) to JSON mapping (.json) and Python classes (.py).
python3 src/scripts/data/parse_ACE_events.py \
    --ace-roles-filepath data/ontology/ace/raw/event_role_ACE.json \
    --ace-entity-filepath data/ontology/ace/raw/event_role_entity_ACE.json \
    --base-entity-filepath data/ontology/ldc/base_entities/base_entities.json \
    --output-filepath data/ontology/ace/processed/ace_roles
"""
import os
import json
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Mapping, Set, Tuple, Union

from src.data.ontology_parse import (
    normalize_to_underscore,
    build_template,
    roles_to_constraint_pairs,
    resolve_entity_dependency,
    generate_cls_definition,
    process_base_entities
)

parser = argparse.ArgumentParser()
parser.add_argument('--ace-roles-filepath',
                    help='Input event roles.', required=True)
parser.add_argument('--ace-entity-filepath',
                    help='Input event roles entities.', required=True)
parser.add_argument('--base-entity-filepath',
                    help='Basic entities mapping in a JSON file.', required=True)
parser.add_argument('--output-filepath', help='Output filepath', required=True)
parser.add_argument('--add-keywords', action='store_true')
parser.add_argument('--add-hierarchy', action='store_true')
parser.add_argument('--add-asserts-child', action='store_true')
parser.add_argument('--add-asserts-parent', action='store_true')
parser.add_argument('--remove-description', action='store_true')
parser.add_argument('--remove-type-annotation', action='store_true')
parser.add_argument('--pure-text-prompt', action='store_true')  # for GPT-3
parser.add_argument(
    '--no-parent-cls',
    action='store_true',
    help='Do not inherit from parent class, but from base class Event.'
)
args = parser.parse_args()


def parse_ACE_events(
    event_role: Dict[str, Dict[str, str]],
    role_constraints: Mapping[str, List[str]],
    base_entities: Mapping[str, Union[str, Dict]],
) -> List[dict]:
    """Parse ACE event roles to code."""

    ret = []
    for name, event in event_role.items():
        # 1. Parse event type (and parent type)
        parent_cls, cur_cls = name.split(":")
        parent_cls = normalize_to_underscore(parent_cls)
        cur_cls = normalize_to_underscore(cur_cls)

        # 2. Process roles to attributes AND fill-in template arguments
        # Classes (e.g. PER) that are required to define downstream event
        if args.remove_description:
            template = None
        else:
            template = build_template(
                event["template"],
                event["roles"],
                event["keywords"] if args.add_keywords else None,
                args
            )

        attribute_constraint_pairs, ontology_cls_dependencies = roles_to_constraint_pairs(
            event["roles"], role_constraints
        )

        # 3. Embed attributes into code class for each event
        # 3.1 Resolve base entity depndencies
        context_code = resolve_entity_dependency(
            base_entities, ontology_cls_dependencies
        )

        # 3.2 Class definition
        event_code = generate_cls_definition(
            cur_cls,
            parent_cls,
            attribute_constraint_pairs,
            docstring=template,
            args=args
        )

        # 4. Add to return list
        ret.append({
            "name": name,
            "parent_cls_name": parent_cls,
            "cur_cls_name": cur_cls,
            "event": event,
            "code": event_code,
            "code_context": context_code,
            "code_with_context": context_code + event_code,
            "ontology_cls_dependencies": ontology_cls_dependencies
        })
    return ret


def parse_hierarchical_ACE_events(
    event_role: Dict[str, Dict[str, str]],
    role_constraints: Mapping[str, List[str]],
    base_entities: Mapping[str, Union[str, Dict]],
):
    # 1. Build a hierarchy of event mapping
    parent_to_child_events = defaultdict(list)
    for event_name, event in event_role.items():
        parent_cls_name, child_cls_name = event_name.split(":")
        parent_cls_name = normalize_to_underscore(parent_cls_name)
        child_cls_name = normalize_to_underscore(child_cls_name)
        event["name"] = event_name
        event["cur_cls_name"] = child_cls_name
        parent_to_child_events[parent_cls_name].append(event)

    # 2. Build parent class
    parent_classes = []
    child_classes = []
    for parent_cls_name, child_events in parent_to_child_events.items():
        # We can use the intersection of all roles as the parent roles
        # parent_roles = set.intersection(*child_roles)
        parent_roles = list(
            sorted(set.union(*[set(child["roles"]) for child in child_events])))

        attribute_constraint_pairs, ontology_cls_dependencies = roles_to_constraint_pairs(
            parent_roles, role_constraints
        )

        cls_context = resolve_entity_dependency(
            base_entities, ontology_cls_dependencies)
        cls_definition = generate_cls_definition(
            parent_cls_name,
            "Event",
            attribute_constraint_pairs,
            docstring=None,
            args=args
        )

        parent_classes.append({
            "name": parent_cls_name,
            "parent_cls_name": "Event",
            "cur_cls_name": parent_cls_name,
            "event": None,
            "code": cls_definition,
            "code_context": cls_context,
            "code_with_context": cls_context + cls_definition,
            "ontology_cls_dependencies": ontology_cls_dependencies
        })

        # 3. Build child classes
        for child_event in child_events:
            child_cls_name = child_event["cur_cls_name"]
            child_roles = child_event["roles"]
            child_attribute_constraint_pairs, child_ontology_cls_dependencies = roles_to_constraint_pairs(
                child_roles, role_constraints
            )

            if args.remove_description:
                template = None
            else:
                template = build_template(
                    child_event["template"],
                    child_roles,
                    child_event["keywords"] if args.add_keywords else None,
                    args
                )

            child_cls_definition = generate_cls_definition(
                child_cls_name,
                parent_cls_name,
                child_attribute_constraint_pairs,
                docstring=template,
                call_super_init=True,
                args=args
            )

            # Use parent class context since child class should also depend on entities that parent class used
            child_cls_context = cls_context + cls_definition + \
                "\n"  # add parent class as context

            child_classes.append({
                "name": child_event["name"],
                "parent_cls_name": parent_cls_name,
                "cur_cls_name": child_cls_name,
                "event": child_event,
                "code": child_cls_definition,
                "code_context": child_cls_context,
                "code_with_context": child_cls_context + child_cls_definition,
                "ontology_cls_dependencies": (ontology_cls_dependencies, child_ontology_cls_dependencies)
            })

    return parent_classes + child_classes


if __name__ == "__main__":
    # 1. Load data
    with open(args.ace_roles_filepath, 'r') as f:
        ace_roles = json.load(f)
    with open(args.ace_entity_filepath, 'r') as f:
        ace_role_entity = json.load(f)
    with open(args.base_entity_filepath, 'r') as f:
        base_entities = json.load(f)

    process_base_entities(base_entities, args)

    # 2. Parse ACE event roles
    if args.add_hierarchy:
        parsed = parse_hierarchical_ACE_events(
            ace_roles,
            ace_role_entity,
            base_entities
        )
    else:
        parsed = parse_ACE_events(
            ace_roles,
            ace_role_entity,
            base_entities
        )

    # 3. Save to file
    print(f"Writing {len(parsed)} events to {args.output_filepath}(.py/.json)")
    # check if output dir exists
    if not os.path.exists(os.path.dirname(args.output_filepath)):
        os.makedirs(os.path.dirname(args.output_filepath))

    with open(args.output_filepath + ".json", 'w') as f:
        json.dump(parsed, f, indent=4)

    with open(args.output_filepath + ".py", 'w') as f:
        f.write(base_entities["header"] + "\n")
        for entity, code in base_entities["base_entities"].items():
            f.write(code + "\n")
        for parsed_event in parsed:
            f.write(parsed_event["code"])
