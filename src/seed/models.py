from dataclasses import dataclass


@dataclass
class Sentence:
    """
    A sentence illustrating a sense of a lemma.
    """

    sentence: str
    word_offsets: list[tuple[int, int]]


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
class Sense:
    """
    A particular sense of a lemma.
    """

    sense_order: int
    definition: str
    sentences: list[Sentence]
