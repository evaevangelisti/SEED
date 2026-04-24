import json
from pathlib import Path
from typing import Any

from tqdm import tqdm


class Mapper:
    """ """

    def __init__(
        self,
        input_path: Path,
    ):
        """
        Initialize the Mapper with the path to the input Wiktextract JSONL file.

        Args:
            input_path (Path): Path to the Wiktextract JSONL file containing lemmas and senses.
        """
        self._input_path: Path = input_path

    @staticmethod
    def _safe_load(
        string: str,
    ) -> dict[str, Any]:
        """
        Safely load a JSON string, returning an empty dictionary if parsing fails.

        Args:
            string (str): The JSON string to parse.

        Returns:
            dict[str, Any]: The parsed JSON object, or an empty dictionary if parsing fails.
        """
        try:
            return json.loads(string)
        except json.JSONDecodeError:
            return {}

    def associate_translations(
        self,
        mappings_path: Path,
    ) -> list[dict[str, Any]]:
        """
        Associate translations with senses based on the provided mappings.

        Args:
            mappings_path (Path): Path to the mappings JSONL file.

        Returns:
            list[dict[str, Any]]: A list of lemma entries with associated translations for each sense, based on the provided mappings.
        """
        lemmas: list[dict[str, Any]] = []

        with mappings_path.open(encoding="utf-8") as file:
            mappings: dict[str, dict[str, str]] = json.load(file)

        with (
            self._input_path.open(encoding="utf-8") as file,
            tqdm(
                desc="Associating translations",
                unit=" lines",
            ) as pbar,
        ):
            for line in file:
                lemma: dict[str, Any] = self._safe_load(line)
                if not lemma:
                    pbar.update(1)
                    continue

                lemma_id: str = lemma.get("id", "")
                mapping: dict[str, str] = mappings.get(lemma_id, {}) if mappings else {}

                translation_map: dict[str, dict[str, list[str]]] = lemma.get(
                    "translations",
                    {},
                )

                translation_keys: list[str] = [
                    key for key in translation_map.keys() if isinstance(key, str)
                ]

                senses: list[dict[str, Any]] = []

                for i, sense in enumerate(lemma.get("senses", []), start=1):
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
                        sense["translations"] = translations

                    senses.append(sense)

                lemma["senses"] = senses
                lemmas.append(lemma)

                pbar.update(1)

        return lemmas

    def associate_wordnet_synset_ids(
        self,
        mappings_path: Path,
    ) -> list[dict[str, Any]]:
        """
        Associate WordNet synset IDs with senses based on the provided mappings.

        Args:
            mappings_path (Path): Path to the mappings JSONL file.

        Returns:
            list[dict[str, Any]]: A list of lemma entries with associated WordNet synset IDs for each sense, based on the provided mappings.
        """
        lemmas: list[dict[str, Any]] = []

        with mappings_path.open(encoding="utf-8") as file:
            mappings: dict[str, list[str]] = json.load(file)

        with (
            self._input_path.open(encoding="utf-8") as file,
            tqdm(
                desc="Associating WordNet synset IDs",
                unit=" lines",
            ) as pbar,
        ):
            for line in file:
                lemma: dict[str, Any] = self._safe_load(line)
                if not lemma:
                    pbar.update(1)
                    continue

                lemma_id: str = lemma.get("id", "")
                wordnet_synset_ids: list[str] = (
                    mappings.get(lemma_id, []) if mappings else []
                )

                senses: list[dict[str, Any]] = []

                for i, sense in enumerate(lemma.get("senses", [])):
                    wordnet_synset_id: str | None = (
                        wordnet_synset_ids[i] if i < len(wordnet_synset_ids) else None
                    )

                    if wordnet_synset_id:
                        sense["wordnet_synset_id"] = wordnet_synset_id

                    senses.append(sense)

                lemma["senses"] = senses
                lemmas.append(lemma)

                pbar.update(1)

        return lemmas
