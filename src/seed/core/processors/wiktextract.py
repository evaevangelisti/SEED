import gzip
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from tqdm import tqdm

from ...models import Example, Quotation, Sense, Sentence, Translation
from .base import Processor


class WiktextractProcessor(Processor):
    """
    Processor for Wiktextract JSONL files.
    """

    _YEAR_PATTERN: re.Pattern = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")

    def __init__(
        self,
        minimum_year: int = datetime.now().year - 25,
        maximum_year: int = datetime.now().year,
    ):
        """
        Initialize the Wiktextract processor.

        Args:
            minimum_year (int): Minimum year for filtering example sentences.
            maximum_year (int): Maximum year for filtering example sentences.
        """
        self._minimum_year: int = minimum_year
        self._maximum_year: int = maximum_year

    def _extract_year(
        self,
        string: str,
    ) -> int | None:
        """
        Extract year from a string.

        Args:
            string (str): Input string.

        Returns:
            int | None: Extracted year or None if not found.
        """
        if not string:
            return None

        year_match: re.Match | None = self._YEAR_PATTERN.search(string)
        if not year_match:
            return None

        return int(year_match.group(0))

    def _extract_sentences(
        self,
        raw_sentences: list[dict[str, Any]],
    ) -> list[Sentence]:
        """
        Extract example sentences from Wiktextract examples.

        Args:
            raw_sentences (list[dict[str, Any]]): List of raw example sentences.

        Returns:
            list[Sentence]: List of extracted Sentence objects.
        """
        sentences: list[Sentence] = []

        for example in raw_sentences:
            text: str | None = example.get("text")
            if not text:
                continue

            sentence: str = text.strip()

            reference: str | None = example.get("ref")
            if reference:
                year: int | None = self._extract_year(reference)
                if year is None:
                    continue

                if self._minimum_year <= year <= self._maximum_year:
                    sentences.append(
                        Quotation(
                            sentence=sentence,
                            reference=reference.strip(),
                        )
                    )
            elif example.get("type", "") == "example":
                sentences.append(Example(sentence=sentence))

        return sentences

    def _extract_senses(
        self,
        raw_senses: list[dict[str, Any]],
    ) -> list[Sense]:
        """
        Extract senses from a Wiktextract entry.

        Args:
            raw_senses (list[dict[str, Any]]): List of raw senses.

        Returns:
            list[Sense]: List of extracted Sense objects.
        """
        senses: list[Sense] = []

        for i, sense in enumerate(raw_senses, start=1):
            glosses: list[str] = sense.get("glosses", [])

            if glosses:
                definition: str = glosses[-1].strip()

                if definition:
                    sentences: list[Sentence] = self._extract_sentences(
                        sense.get("examples", []),
                    )

                    if sentences:
                        senses.append(
                            Sense(
                                sense_order=i,
                                definition=definition,
                                sentences=sentences,
                            )
                        )

        return senses

    def _extract_translations(
        self,
        raw_translations: list[dict[str, Any]],
    ) -> dict[str, list[Translation]]:
        """
        Extract translations from a Wiktextract entry.

        Args:
            raw_translations (list[dict[str, Any]]): List of raw translations.

        Returns:
            dict[str, list[Translation]]: Dictionary mapping sense definitions to lists of Translations.
        """
        translations: dict[str, list[Translation]] = defaultdict(list)

        for translation in raw_translations:
            sense: str | None = translation.get("sense")
            if not sense:
                continue

            sense = sense.strip()

            word: str | None = translation.get("word")
            if not word:
                continue

            word = word.strip().lower()

            language: str | None = translation.get("lang")
            if not language:
                continue

            language = language.strip().lower()

            languages: set[str] = {
                translation.language for translation in translations[sense]
            }

            if language not in languages:
                translations[sense].append(
                    Translation(
                        translation=word,
                        language=language,
                    )
                )

        return translations

    def extract_lemmas(
        self,
        input_path: Path,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Extract lemmas and their translations from the Wiktextract JSONL file.

        Args:
            input_path (Path): Path to the Wiktextract JSONL file.

        Returns:
            Generator[dict[str, Any], None, None]: Generator yielding lemmas with their details.
        """
        with (
            gzip.open(input_path, "rt", encoding="utf-8") as file,
            tqdm(
                desc="Extracting",
                unit=" lines",
            ) as pbar,
        ):
            for line in file:
                entry: dict[str, Any] = json.loads(line)

                language: str | None = entry.get("lang_code") or entry.get("lang")
                if language and language.lower() in ("en", "english"):
                    lemma: str | None = entry.get("word")
                    if lemma:
                        lemma = lemma.strip()

                        etymology: str | None = entry.get("etymology_text")
                        pos: str | None = entry.get("pos")

                        senses: list[Sense] = self._extract_senses(
                            entry.get("senses", []),
                        )

                        if senses:
                            translations: dict[str, list[Translation]] = (
                                self._extract_translations(
                                    entry.get("translations", []),
                                )
                            )

                            yield {
                                "lemma": lemma,
                                "etymology": etymology,
                                "pos": pos,
                                "senses": senses,
                                "translations": translations,
                            }

                pbar.update(1)

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

    def _build_mappings(
        self,
        mappings_path: Path,
    ) -> dict[tuple[str, str, str], dict[str, str]]:
        """
        Build a mapping dictionary from the mappings JSONL file.

        Args:
            mappings_path (Path): Path to the mappings JSONL file.

        Returns:
            dict[tuple[str, str, str], dict[str, str]]: A dictionary mapping (lemma, etymology, pos) to a mapping of sense index to translation letter.
        """
        mappings: dict[tuple[str, str, str], dict[str, str]] = {}

        with mappings_path.open(encoding="utf-8") as file:
            for line in file:
                entry = self._safe_load(line)
                if not entry:
                    continue

                key: tuple[str, str, str] = (
                    entry.get("lemma", ""),
                    entry.get("etymology", ""),
                    entry.get("pos", ""),
                )

                if key in mappings:
                    mappings[key] = {}
                    continue

                raw_mapping: dict[str, str] | None = entry.get("mapping")
                if isinstance(raw_mapping, dict):
                    mappings.setdefault(key, raw_mapping)

        return mappings

    def associate_translations(
        self,
        input_path: Path,
        mappings_path: Path,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Associate translations with senses based on the provided mappings.

        Args:
            input_path (Path): Path to the Wiktextract JSONL file.
            mappings_path (Path): Path to the mappings JSONL file.

        Returns:
            Generator[dict[str, Any], None, None]: Generator yielding lemmas with associated translations.
        """
        mappings: dict[tuple[str, str, str], dict[str, str]] = self._build_mappings(
            mappings_path,
        )

        with (
            input_path.open(encoding="utf-8") as file,
            tqdm(
                desc="Associating",
                unit=" lines",
            ) as pbar,
        ):
            for line in file:
                input_entry: dict[str, Any] = self._safe_load(line)
                if not input_entry:
                    pbar.update(1)
                    continue

                lemma: str = input_entry.get("lemma", "")
                etymology: str = input_entry.get("etymology", "")
                pos: str = input_entry.get("pos", "")

                translations_map: dict[str, list[dict[str, str]]] = input_entry.get(
                    "translations",
                    {},
                )

                translations_keys: list[str] = [
                    key for key in translations_map.keys() if isinstance(key, str)
                ]

                key: tuple[str, str, str] = (lemma, etymology, pos)
                mapping: dict[str, str] = mappings.get(key, {}) if mappings else {}

                senses: list[dict[str, Any]] = []

                for i, sense in enumerate(input_entry.get("senses", []), start=1):
                    translations: list[Translation] = []

                    letter: str | None = (
                        mapping.get(str(i)) if isinstance(mapping, dict) else None
                    )

                    raw_translations: list[dict[str, str]] | None = None
                    if letter and len(letter) == 1 and letter.isalpha():
                        index: int = ord(letter.upper()) - ord("A")
                        if 0 <= index < len(translations_keys):
                            raw_translations = translations_map.get(
                                translations_keys[index]
                            )

                    if raw_translations:
                        for raw_translation in raw_translations:
                            if not isinstance(raw_translation, dict):
                                continue

                            translation: str | None = raw_translation.get(
                                "translation",
                            )

                            language: str | None = raw_translation.get(
                                "language",
                            )

                            if translation and language:
                                translations.append(
                                    Translation(
                                        translation=translation,
                                        language=language,
                                    )
                                )

                    senses.append(
                        {
                            "sense_order": sense.get("sense_order"),
                            "definition": sense.get("definition"),
                            "sentences": sense.get("sentences"),
                            "translations": translations,
                        }
                    )

                yield {
                    "lemma": lemma,
                    "etymology": etymology,
                    "pos": pos,
                    "senses": senses,
                }

                pbar.update(1)
