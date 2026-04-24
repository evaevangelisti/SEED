from collections import defaultdict
from typing import Any

from nltk.corpus import wordnet as wn
from tqdm import tqdm

from ..models import POS, WordNetSense
from .base import Processor


class WordNetProcessor(Processor):
    """
    Processor for WordNet.
    """

    def __init__(
        self,
        allowed_pos_tags: set[POS] | None = None,
    ):
        """
        Initialize the WordNetProcessor.

        Args:
            allowed_pos_tags (set[POS] | None): A set of allowed parts of speech to filter the synsets. If None, all parts of speech are included.
        """
        self._allowed_pos_tags: set[POS] | None = allowed_pos_tags

    def extract_lemmas(
        self,
    ) -> list[dict[str, Any]]:
        """
        Extract lemmas, their parts of speech, and their senses from WordNet.

        Returns:
            list[dict[str, Any]]: A list of dictionaries, each containing a lemma, its part of speech, and a list of its senses (definitions and synonyms).
        """
        records: dict[tuple[str, POS], list[WordNetSense]] = defaultdict(list)

        for synset in tqdm(
            wn.all_synsets(),
            desc="Extracting lemmas from WordNet",
            unit=" synset",
        ):
            pos_tag: POS | None = POS.from_wordnet(synset.pos())
            if pos_tag is not None and (
                self._allowed_pos_tags is None or pos_tag in self._allowed_pos_tags
            ):
                synset_id: str = synset.name()
                definition: str = synset.definition()

                lemmas: list[str] = [lemma.name() for lemma in synset.lemmas()]
                for lemma in lemmas:
                    synonyms: list[str] = [
                        lemma_name for lemma_name in lemmas if lemma_name != lemma
                    ]

                    senses: list[WordNetSense] = records[(lemma, pos_tag)]

                    existing_sense: WordNetSense | None = next(
                        (sense for sense in senses if sense.id == synset_id),
                        None,
                    )

                    if existing_sense:
                        existing_sense.synonyms = list(
                            set(existing_sense.synonyms) | set(synonyms)
                        )
                    else:
                        senses.append(
                            WordNetSense(
                                id=synset_id,
                                definition=definition,
                                synonyms=synonyms,
                            )
                        )

        return [
            {
                "lemma": lemma,
                "pos": pos_tag.value,
                "senses": senses,
            }
            for (lemma, pos_tag), senses in records.items()
            if senses
        ]
