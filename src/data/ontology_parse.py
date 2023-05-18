import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

CLS_TO_ASSERT_STATEMENTS = {
    "Movement": [
        (
            ["Transport"],
            "assert not bool(set(self.origin) & set(self.destination)), \"Origin and destination cannot be the same.\"",
        )
    ],
    "Personnel": [],
    "Conflict": [
        (
            ["Attack"],
            "assert not bool(set(self.attacker) & set(self.victim)), \"Attacker and victim cannot be the same.\"",
        )
    ],
    "Contact": [],
    "Life": [],
    "Transaction": [
        (
            ["Transfer_Money"],
            "assert not bool(set(self.giver) & set(self.recipient)), \"Giver and receiver cannot be the same.\"",
        ),
        (
            ["Transfer_Ownership"],
            "assert not bool(set(self.buyer) & set(self.seller)), \"Buyer and seller cannot be the same.\""
        ),
    ],
    "Business": [],
    "Justice": [
        (
            ["Sue"],
            "assert not bool(set(self.plaintiff) & set(self.defendant)), \"Plaintiff and defendant cannot be the same.\""
        ),
        (
            ["Sue", "Trial_Hearing", "Charge_Indict",
                "Convict", "Sentence", "Pardon", "Acquit"],
            "assert not bool(set(self.adjudicator) & set(self.defendant)), \"Adjudicator and defendant cannot be the same.\""
        ),
        (
            ["Trial_Hearing", "Charge_Indict"],
            "assert not bool(set(self.prosecutor) & set(self.defendant)), \"Prosecutor and defendant cannot be the same.\""
        ),
        (
            ["Release_Parole"],
            "assert not bool(set(self.entity) & set(self.person)), \"Entity and person cannot be the same.\""
        ),
        (
            ["Fine"],
            "assert not bool(set(self.adjudicator) & set(self.entity)), \"Adjudicator and entity cannot be the same.\""
        ),
        (
            ["Appeal"],
            "assert not bool(set(self.plaintiff) & set(self.adjudicator)), \"Plaintiff and adjudicator cannot be the same.\""
        ),
        (
            ["Arrest_Jail", "Execute", "Extradite"],
            "assert not bool(set(self.agent) & set(self.person)), \"Agent and person cannot be the same.\""
        ),
        (
            ["Extradite"],
            "assert not bool(set(self.origin) & set(self.destination)), \"Origin and destination cannot be the same.\""
        ),
    ],
}


def normalize_to_underscore(string: str) -> str:
    """Normalize string to only use underscore."""
    return string.replace(" ", "_").replace("-", "_")


def process_base_entities(base_entities: Dict, args=argparse.Namespace()) -> Dict:
    if args.pure_text_prompt:
        base_entities["header"] = "Description of base entity types:"
        for base_entity in base_entities["base_entities"]:
            # pure text prompt, only keep the class description
            base_entities["base_entities"][base_entity] = f"{base_entity}: " + \
                base_entities["base_entities"][base_entity].split("\"\"\"")[1]
    return base_entities


def resolve_entity_dependency(base_entities: Dict, ontology_cls_dependencies: List):
    context_code = base_entities["header"]
    for base_entity in ontology_cls_dependencies:
        context_code += "\n" + base_entities["base_entities"][base_entity]
    context_code += "\n"
    return context_code


def generate_cls_definition(
    cur_cls,
    parent_cls,
    attribute_constraint_pairs,
    docstring=None,
    call_super_init=False,
    args=argparse.Namespace()
):

    if args.pure_text_prompt:
        assert not args.remove_type_annotation
        _parent_cls_str = "" if args.no_parent_cls else f" (Parent type: {parent_cls})"
        event_def = f"\nRole definition of event type {cur_cls}{_parent_cls_str}:\n"
        for i, _tup in enumerate(attribute_constraint_pairs):
            attribute, constraints = _tup
            constraints = constraints.replace(
                "List[", "").replace("]", "").replace(" | ", " or ")
            # NOTE: order of attributes is sorted by attribute name
            event_def += f"{i+1}. {attribute} (need to be one of {constraints})\n"
        event_def += "Multiple entities can be extracted for the same role, each entity is a double-quote enclosed string.\n"
        event_def += "Each extracted entity should look like: (Base Entity Type) \"content of extracted string\"\n"
        event_def += "If entity is not present in the text, write: () \"\"\n"
        event_def += "Different entities are delimited by a comma.\n"
        event_def += f"In this event: {docstring.strip()}\n\n"
        return event_def

    _parent_cls_str = "(Event)" if args.no_parent_cls else f"({parent_cls})"
    event_code = f"class {cur_cls}{_parent_cls_str}:\n"
    if docstring is not None:
        event_code += f"    \"\"\"{docstring}\"\"\"\n"

    event_code += (
        f"    def __init__(\n"
        f"        self,\n"
    )

    for attribute, constraint in attribute_constraint_pairs:
        if args.remove_type_annotation:
            event_code += f"        {attribute} = [],\n"
        else:
            event_code += f"        {attribute}: {constraint} = [],\n"
    event_code += "    ):\n"

    # 3.3 Assign class attributes
    if not call_super_init:
        for idx, attribute_constraint_pair in enumerate(attribute_constraint_pairs):
            attribute, constraint = attribute_constraint_pair
            event_code += f"        self.{attribute} = {attribute}\n"
        event_code += "\n"
    else:
        event_code += (
            f"        super().__init__(\n"
        )
        for idx, attribute_constraint_pair in enumerate(attribute_constraint_pairs):
            attribute, constraint = attribute_constraint_pair
            event_code += f"            {attribute}={attribute},\n"
        event_code += "        )\n\n"

    if args.add_asserts_child and parent_cls != "Event":
        # Add only asserts for current class
        cur_parent_asserts = CLS_TO_ASSERT_STATEMENTS[parent_cls]
        asserts_to_add = [
            assert_statement
            for child_type, assert_statement in cur_parent_asserts
            if cur_cls in child_type
        ]
        for assert_statement in asserts_to_add:
            event_code += f"        {assert_statement}\n"
        event_code += "\n"
    if args.add_asserts_parent and parent_cls == "Event":
        # Add all asserts for parent class
        cur_parent_asserts = CLS_TO_ASSERT_STATEMENTS[cur_cls]
        asserts_to_add = [
            assert_statement for child_type, assert_statement in cur_parent_asserts
        ]
        for assert_statement in asserts_to_add:
            event_code += f"        {assert_statement}\n"
        event_code += "\n"

    return event_code


def roles_to_constraint_pairs(roles: List[str], role_constraints) -> List[Tuple[str, str]]:
    ontology_cls_dependencies = set()
    attribute_constraint_pairs: Set[Tuple[str, List[str]]] = set()

    for idx, role in enumerate(roles):
        # 2.1 Process constraints for this role
        # Constraints: xxx | yyy | None
        assert role in role_constraints, f"Role {role} not found in constraints."
        constraints = sorted(role_constraints.get(role))
        # Add constraint role as dependency to add
        ontology_cls_dependencies.update(constraints)
        # constraints = "Union[" + ", ".join(constraints) + ", None]"
        constraints = "List[" + " | ".join(constraints) + "]"

        # 2.2 Covert role to class attribute
        attribute = role.lower()

        # 2.3 Add to attribute list
        attribute_constraint_pairs.add((attribute, constraints))

    # sort by attribute name
    attribute_constraint_pairs = sorted(
        list(attribute_constraint_pairs), key=lambda x: x[0])
    ontology_cls_dependencies = sorted(list(ontology_cls_dependencies))
    return attribute_constraint_pairs, ontology_cls_dependencies


def build_template(
    event_template: str,
    roles: List[str],
    keywords: List[str] = None,
    args=argparse.Namespace()
):
    # TODO: this is kind of ACE05 specific for now
    role_tot_count = Counter(roles)
    role_counter = defaultdict(int)

    template = event_template
    for idx, role in enumerate(roles):
        # 2.2 Covert role to attribute AND Process template arguments
        attribute = role.lower()
        if role_tot_count[role] > 1:
            # multiple same attributes in roles (e.g., [Entity, Entity, Place])
            # we still only use one attribute (e.g., self.entity)
            # which is a list of all related entities
            # here, we hint the model to extract entities in template
            template = template.replace(
                f"<arg{idx+1}>",
                f"self.{attribute}[{role_counter[role]}]" if not args.pure_text_prompt else f"[{attribute}{role_counter[role]+1}]"
            )
            role_counter[role] += 1
        else:
            # unique attribute
            template = template.replace(
                f"<arg{idx+1}>",
                f"self.{attribute}" if not args.pure_text_prompt else f"[{attribute}]"
            )

    if keywords is not None:
        template = f"\n    {template}.\n    Event keywords: {', '.join(keywords)}.\n    "
    else:
        template += "."
    return template
