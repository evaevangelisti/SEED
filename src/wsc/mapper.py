import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .config import WIKTEXTRACT_TRANSLATIONS_PATH
from .models import WiktionaryLemma, WiktionarySense


class Mapper:
    """
    Mapper class responsible for associating translations and WordNet synset IDs with senses based on provided mappings.
    """

    def __init__(
        self,
        lemmas: list[WiktionaryLemma],
    ):
        """
        Initialize the Mapper with the path to the input Wiktextract JSONL file.

        Args:
            lemmas (list[WiktionaryLemma]): A list of WiktionaryLemma instances to be processed.
        """
        self._lemmas: list[WiktionaryLemma] = lemmas

    def associate_translations(
        self,
        mappings_path: Path,
    ) -> list[WiktionaryLemma]:
        """
        Associate translations with senses based on the provided mappings.

        Args:
            mappings_path (Path): Path to the mappings JSONL file.

        Returns:
            list[WiktionaryLemma]: A list of WiktionaryLemma instances with associated translations for each sense, based on the provided mappings.
        """
        translations_maps: dict[str, Any] = {}

        with WIKTEXTRACT_TRANSLATIONS_PATH.open(encoding="utf-8") as file:
            for line in file:
                record: dict[str, Any] = json.loads(line)
                translations_maps[record["id"]] = record.get("translations", {})

        with mappings_path.open(encoding="utf-8") as file:
            mappings: dict[str, dict[str, str]] = json.load(file)

        for lemma in tqdm(self._lemmas, desc="Associating translations", unit=" lemma"):
            lemma_id: str = lemma.id
            mapping: dict[str, str] = mappings.get(lemma_id, {}) if mappings else {}

            translation_map: dict[str, dict[str, list[str]]] = translations_maps.get(
                lemma_id,
                {},
            )

            translation_keys: list[str] = [
                key for key in translation_map.keys() if isinstance(key, str)
            ]

            senses: list[WiktionarySense] = []

            for i, sense in enumerate(lemma.senses, start=1):
                translations: dict[str, list[str]] = {}

                mapped: str | None = mapping.get(f"F{i}")

                raw_translations: dict[str, list[str]] | None = None
                if mapped and mapped.startswith("S"):
                    try:
                        index: int = int(mapped[1:]) - 1
                        if 0 <= index < len(translation_keys):
                            raw_translations = translation_map.get(
                                translation_keys[index]
                            )
                    except ValueError:
                        pass

                if raw_translations:
                    for language, words in raw_translations.items():
                        if not isinstance(words, list):
                            continue

                        translations.setdefault(language, [])
                        for word in words:
                            if word not in translations[language]:
                                translations[language].append(word)

                if translations:
                    sense.translations = translations

                senses.append(sense)

            lemma.senses = senses

        return self._lemmas

    def associate_wordnet_synset_ids(
        self,
        mappings_path: Path,
    ) -> list[WiktionaryLemma]:
        """
        Associate WordNet synset IDs with senses based on the provided mappings.

        Args:
            mappings_path (Path): Path to the mappings JSONL file.

        Returns:
            list[WiktionaryLemma]: A list of WiktionaryLemma instances with associated WordNet synset IDs for each sense, based on the provided mappings.
        """
        with mappings_path.open(encoding="utf-8") as file:
            mappings: dict[str, list[str]] = json.load(file)

        for lemma in tqdm(
            self._lemmas, desc="Associating WordNet synset IDs", unit=" lemma"
        ):
            lemma_id: str = lemma.id
            wordnet_synset_ids: list[str] = (
                mappings.get(lemma_id, []) if mappings else []
            )

            senses: list[WiktionarySense] = []

            for i, sense in enumerate(lemma.senses):
                wordnet_synset_id: str | None = (
                    wordnet_synset_ids[i] if i < len(wordnet_synset_ids) else None
                )

                if wordnet_synset_id:
                    sense.wordnet_synset_id = wordnet_synset_id

                senses.append(sense)

            lemma.senses = senses

        return self._lemmas
