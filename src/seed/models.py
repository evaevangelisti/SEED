from dataclasses import dataclass, field


@dataclass
class Sentence:
    """
    A sentence illustrating a sense of a lemma.
    """

    sentence: str


@dataclass
class Example(Sentence):
    """
    An example sentence illustrating a sense of a lemma.
    """

    pass


@dataclass
class Quotation(Sentence):
    """
    A quotation sentence illustrating a sense of a lemma.
    """

    reference: str


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
    sentences: list[Sentence]


@dataclass
class Lemma:
    """
    A lemma with its senses.
    """

    lemma: str
    etymology: str | None = None
    pos: str | None = None
    senses: list[Sense] = field(default_factory=list)
