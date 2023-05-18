"""
Convert LDC base annotation type to JSON mapping (.json) and Python class (.py).
python3 src/scripts/data/build-ldc-base-entities.py data/ontology/ldc/base_entities/base_entities
"""
import re
import pandas as pd
import autopep8
import argparse
import json
from typing import List, Dict, Tuple


def capitalize_initial(t):
    if len(t) < 2:
        return t.upper()
    return t[0].upper() + t[1:]


def cw2us(x):
    """capwords to underscore notation"""
    return re.sub(r'(?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])',
                  r"_\g<0>", x).lower()


def us2mc(x):
    """Convert underscore to mixed-case notation."""
    x = x.replace("-", "")
    x = x.replace(" ", "_")
    text = re.sub(r'_([a-z])', lambda m: (m.group(1).upper()), x)
    return capitalize_initial(text)


def format_desc(desc):
    desc = capitalize_initial(" ".join(desc.strip().split()))
    if len(desc) > 0 and not desc.endswith("."):
        desc += "."
    return desc


entity_type_mapping = {
    # Get from OneIE
    # https://github.com/GAIA-IE/oneie/blob/master/oneie/convert.py
    'URL': 'URL',
    'TME': 'Time',
    'TTL': 'Title',
    'MON': 'Money',
    'PER': 'Person',
    'WEA': 'Weapon',
    'VEH': 'Vehicle',
    'LOC': 'Location',
    'FAC': 'Facility',
    'ORG': 'Organization',
    'VAL': 'NumericalValue',
    'GPE': 'GeopoliticalEntity',
    'CRM': 'Crime',
    'LAW': 'Law',
    'BAL': 'Ballot',
    'COM': 'Commodity',
    'SID': 'Sides',

    # Manually added - These are not used in ACE05
    'ABS': 'Abstract',
    'AML': 'Animal',
    'BOD': 'Body',
    'INF': 'Information',
    'MHI': 'MentalHealthIssue',
    'NAT': 'NaturalMaterials',
    'PLA': 'Plant',
    'PTH': 'InfectiousAgent',
    'RES': 'VotingResult',
    'SEN': 'JusticeSentence',
}


ldc_anno_type = """
Type	Output Value for Type	Definition
ABS	abs	Abstract, non-tangible artifacts such as software (e.g., programs, tool kits, apps, e-mail), measureable intellectual property, contracts, etc. (nb: does not include laws, which are LAW type)
AML	aml	Animal, a non-human living organism which feeds on organic matter, typically having specialized sense organs and a nervous system and able to respond rapidly to stimuli
BAL	bal	A ballot for an election, either the paper ballot used for voting, including both physical paper ballots and also the slate of candidates and ballot questions
BOD	bod	An identifiable, living part of a human's or animal's body, such as a eye, ear, neck, leg, etc.
COM	com	A tangible product or article of trade for which someone pays or barters, or more generally, an artifact or a thing
FAC	fac	A functional, primarily man-made structure. Facilities are artifacts falling under the domains of architecture and civil engineering, including more temporary human constructs, such as police lines and checkpoints.
GPE	gpe	Geopolitical entities such as countries, provinces, states, cities, towns, etc. GPEs are composite entities, consisting of a physical location, a government, and a population. All three of these elements must be present for an entity to be tagged as a GPE. A GPE entity may be a single geopolitical entity or a group.
INF	inf	An information object such as a field of study or topic of communication, including thoughts, opinions, etc.
LAW	law	A law, an act that is voted on by either a legislative body or an electorate, such as a law, referendum, act, regulation, statute, ordinance, etc.
LOC	loc	Geographical entities such as geographical areas and landmasses, bodies of water
MHI	mhi	Any medical condition or health issue, to include everything from disease to broken bones to fever to general ill health, medical errors, even natural causes
MON	mon	A monetary payment. The extent of a Money mention includes modifying quantifiers, the amount, and the currency unit, all of which can be optional.
NAT	nat	Valuable materials or substances, such as minerals, forests, water, and fertile land, that are not man-made, occur naturally within the environment and can be used for economic gain
ORG	org	Corporations, agencies, and other groups of people defined by an established organizational structure. An ORG entity may be a single organization or a group. A key feature of an ORG is that it can change members without changing identity.
PER	per	Person entities are limited to humans. A PER entity may be a single person or a group.
PLA	pla	Plants/flora as well as edible fungi such as mushrooms; multicellular living organisms, typically growing in the earth and  lacking the power of locomotion, ex. grass and crops such as wheat, beans, fruit, etc.
PTH	pth	An infectious microorganism or agent, such as a virus, bacterium, protozoan, prion, viroid, or fungus
RES	res	The results of a voting event.  This will cover general results as well as counted results.
SEN	sen	The judicial or court sentence in a Justice event, the punishment a judge gives to a defendant found guilty of a crime
SID	sid	The different sides of a conflict, such as  philosophical, cultural, ideological, religious, political, guiding philosophical movement or group orientation.  This will encompass sides of the battle/conflict, sports fans when salient, and other such affiliations, in addition to religions, political parties, and other philosophies.
TTL	ttl	A person’s title or job role
VAL	val	A numerical value or non-numerical value such as an informational property such as color or make or URL
VEH	veh	A physical device primarily designed to move an object from one location to another, by (for example) carrying, flying, pulling, or pushing the transported object. Vehicle entities may or may not have their own power source.
WEA	wea	A physical device that is primarily used as an instrument for physically harming or destroying entities
"""

if __name__ == "__main__":
    # Manually Defined For Class Name to include semantic information
    ldc_anno_type = [line.split("\t")
                     for line in ldc_anno_type.split("\n") if len(line)]
    ldc_anno_type = pd.DataFrame(ldc_anno_type[1:], columns=ldc_anno_type[0])

    parser = argparse.ArgumentParser()
    parser.add_argument('output_filepath')
    parser.add_argument('--no-init', action='store_true')
    parser.add_argument('--informative-name', action='store_true')

    args = parser.parse_args()

    def build_ldc_anno_cls(anno_type: str, anno_desc: str):
        entity_name = us2mc(anno_type)
        if args.informative_name:
            entity_name = entity_type_mapping[anno_type]

        ret = (
            f"class {entity_name}(Entity):\n"
            f"    \"\"\"{anno_desc}\"\"\"\n"
        )

        if args.no_init:
            ret += (
                f"    pass\n"
            )
        else:
            ret += (
                f"    def __init__(self, name: str):\n"
                f"        super().__init__(name=name)\n"
            )

        return ret, entity_name

    base_entities: List[dict] = []
    # Build LDC Base Class
    for _, row in ldc_anno_type.iterrows():
        entity_cls, entity_name = build_ldc_anno_cls(
            row["Type"], row["Definition"])
        base_entities.append({
            "name": entity_name,
            "code": entity_cls
        })

    root_entity = """
    from typing import List

    class Entity:
        def __init__(self, name: str):
            self.name = name

    class Event:
        def __init__(self, name: str):
            self.name = name
    """
    CODE = root_entity + "".join([entity["code"] for entity in base_entities])
    CODE = autopep8.fix_code(CODE)

    print(
        f"Writing {len(base_entities)} classes to {args.output_filepath} (.py/.json)"
    )
    with open(args.output_filepath + ".py", "w") as f:
        f.write(CODE)

    with open(args.output_filepath + ".json", "w") as f:
        f.write(json.dumps(
            {
                "base_entities": {entity["name"]: entity["code"] for entity in base_entities},
                "header": root_entity
            },
            indent=4
        ))
