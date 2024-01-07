
from typing import List


class Entity:
    def __init__(self, name: str):
        self.name = name


class Event:
    def __init__(self, name: str):
        self.name = name


class ABS(Entity):
    """Abstract, non-tangible artifacts such as software (e.g., programs, tool kits, apps, e-mail), measureable intellectual property, contracts, etc. (nb: does not include laws, which are LAW type)"""

    def __init__(self, name: str):
        super().__init__(name=name)


class AML(Entity):
    """Animal, a non-human living organism which feeds on organic matter, typically having specialized sense organs and a nervous system and able to respond rapidly to stimuli"""

    def __init__(self, name: str):
        super().__init__(name=name)


class BAL(Entity):
    """A ballot for an election, either the paper ballot used for voting, including both physical paper ballots and also the slate of candidates and ballot questions"""

    def __init__(self, name: str):
        super().__init__(name=name)


class BOD(Entity):
    """An identifiable, living part of a human's or animal's body, such as a eye, ear, neck, leg, etc."""

    def __init__(self, name: str):
        super().__init__(name=name)


class COM(Entity):
    """A tangible product or article of trade for which someone pays or barters, or more generally, an artifact or a thing"""

    def __init__(self, name: str):
        super().__init__(name=name)


class FAC(Entity):
    """A functional, primarily man-made structure. Facilities are artifacts falling under the domains of architecture and civil engineering, including more temporary human constructs, such as police lines and checkpoints."""

    def __init__(self, name: str):
        super().__init__(name=name)


class GPE(Entity):
    """Geopolitical entities such as countries, provinces, states, cities, towns, etc. GPEs are composite entities, consisting of a physical location, a government, and a population. All three of these elements must be present for an entity to be tagged as a GPE. A GPE entity may be a single geopolitical entity or a group."""

    def __init__(self, name: str):
        super().__init__(name=name)


class INF(Entity):
    """An information object such as a field of study or topic of communication, including thoughts, opinions, etc."""

    def __init__(self, name: str):
        super().__init__(name=name)


class LAW(Entity):
    """A law, an act that is voted on by either a legislative body or an electorate, such as a law, referendum, act, regulation, statute, ordinance, etc."""

    def __init__(self, name: str):
        super().__init__(name=name)


class LOC(Entity):
    """Geographical entities such as geographical areas and landmasses, bodies of water"""

    def __init__(self, name: str):
        super().__init__(name=name)


class MHI(Entity):
    """Any medical condition or health issue, to include everything from disease to broken bones to fever to general ill health, medical errors, even natural causes"""

    def __init__(self, name: str):
        super().__init__(name=name)


class MON(Entity):
    """A monetary payment. The extent of a Money mention includes modifying quantifiers, the amount, and the currency unit, all of which can be optional."""

    def __init__(self, name: str):
        super().__init__(name=name)


class NAT(Entity):
    """Valuable materials or substances, such as minerals, forests, water, and fertile land, that are not man-made, occur naturally within the environment and can be used for economic gain"""

    def __init__(self, name: str):
        super().__init__(name=name)


class ORG(Entity):
    """Corporations, agencies, and other groups of people defined by an established organizational structure. An ORG entity may be a single organization or a group. A key feature of an ORG is that it can change members without changing identity."""

    def __init__(self, name: str):
        super().__init__(name=name)


class PER(Entity):
    """Person entities are limited to humans. A PER entity may be a single person or a group."""

    def __init__(self, name: str):
        super().__init__(name=name)


class PLA(Entity):
    """Plants/flora as well as edible fungi such as mushrooms; multicellular living organisms, typically growing in the earth and  lacking the power of locomotion, ex. grass and crops such as wheat, beans, fruit, etc."""

    def __init__(self, name: str):
        super().__init__(name=name)


class PTH(Entity):
    """An infectious microorganism or agent, such as a virus, bacterium, protozoan, prion, viroid, or fungus"""

    def __init__(self, name: str):
        super().__init__(name=name)


class RES(Entity):
    """The results of a voting event.  This will cover general results as well as counted results."""

    def __init__(self, name: str):
        super().__init__(name=name)


class SEN(Entity):
    """The judicial or court sentence in a Justice event, the punishment a judge gives to a defendant found guilty of a crime"""

    def __init__(self, name: str):
        super().__init__(name=name)


class SID(Entity):
    """The different sides of a conflict, such as  philosophical, cultural, ideological, religious, political, guiding philosophical movement or group orientation.  This will encompass sides of the battle/conflict, sports fans when salient, and other such affiliations, in addition to religions, political parties, and other philosophies."""

    def __init__(self, name: str):
        super().__init__(name=name)


class TTL(Entity):
    """A person’s title or job role"""

    def __init__(self, name: str):
        super().__init__(name=name)


class VAL(Entity):
    """A numerical value or non-numerical value such as an informational property such as color or make or URL"""

    def __init__(self, name: str):
        super().__init__(name=name)


class VEH(Entity):
    """A physical device primarily designed to move an object from one location to another, by (for example) carrying, flying, pulling, or pushing the transported object. Vehicle entities may or may not have their own power source."""

    def __init__(self, name: str):
        super().__init__(name=name)


class WEA(Entity):
    """A physical device that is primarily used as an instrument for physically harming or destroying entities"""

    def __init__(self, name: str):
        super().__init__(name=name)
