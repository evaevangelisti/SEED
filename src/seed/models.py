from dataclasses import dataclass, field


@dataclass
class Translation:
    """
    A translation of a lemma in a particular sense.
    """

    translation: str
    language: str


@dataclass
class Sense:
    """
    A particular sense of a lemma.
    """

    sense_order: int
    definition: str
    sentences: list[str]
    translations: list[Translation] = field(default_factory=list)


@dataclass
class Lemma:
    """
    A lemma with its senses.
    """

    lemma: str
    senses: list[Sense] = field(default_factory=list)
