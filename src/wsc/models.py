from dataclasses import dataclass
from enum import Enum
from typing import Any


class POS(Enum):
    """
    Part of speech tags.
    """

    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adj"
    ADVERB = "adv"

    @classmethod
    def from_wiktionary(
        cls,
        code: str,
    ) -> "POS | None":
        """
        Convert a Wiktionary POS code to a POS enum member.

        Args:
            code (str): A Wiktionary POS code.

        Returns:
            POS | None: The corresponding POS enum member.
        """
        return {
            "noun": cls.NOUN,
            "verb": cls.VERB,
            "adj": cls.ADJECTIVE,
            "adv": cls.ADVERB,
        }.get(code)

    @classmethod
    def from_wordnet(
        cls,
        code: str,
    ) -> "POS | None":
        """
        Convert a WordNet POS code to a POS enum member.

        Args:
            code (str): A WordNet POS code.

        Returns:
            POS | None: The corresponding POS enum member.
        """
        return {
            "n": cls.NOUN,
            "v": cls.VERB,
            "a": cls.ADJECTIVE,
            "r": cls.ADVERB,
            "s": cls.ADJECTIVE,
        }.get(code)


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
class WiktionarySense:
    """
    A sense of a lemma extracted from Wiktionary.
    """

    definition: str
    sentences: list[Sentence]
    parent_glosses: list[str] | None = None
    translations: dict[str, list[str]] | None = None
    wordnet_synset_id: str | None = None


@dataclass
class WiktionaryLemma:
    """
    A lemma extracted from Wiktionary.
    """

    id: str
    lemma: str
    pos: POS
    senses: list[WiktionarySense]

    @classmethod
    def from_dict(
        cls,
        lemma: dict[str, Any],
    ) -> "WiktionaryLemma":
        """
        Create a WiktionaryLemma instance from a dictionary.

        Args:
            data (dict[str, Any]): A dictionary containing the lemma data.

        Returns:
            WiktionaryLemma: An instance of WiktionaryLemma created from the provided data.
        """
        pos = POS.from_wiktionary(lemma["pos"])
        if pos is None:
            raise ValueError(f"Invalid POS code: {lemma['pos']}")

        return cls(
            id=lemma["id"],
            lemma=lemma["lemma"],
            pos=pos,
            senses=[WiktionarySense(**sense) for sense in lemma.get("senses", [])],
        )


@dataclass
class WordNetSense:
    """
    A sense of a lemma extracted from WordNet.
    """

    id: str
    definition: str
    synonyms: list[str]


@dataclass
class WordNetLemma:
    """
    A lemma extracted from WordNet.
    """

    lemma: str
    pos: POS
    senses: list[WordNetSense]
