import gzip
import json
import re
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from tqdm import tqdm

from ...models import Example, Quotation, Sense, Sentence
from .base import Processor


class WiktextractProcessor(Processor):
    """
    Processor for Wiktextract JSONL files.
    """

    _YEAR_PATTERN: re.Pattern = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")

    def __init__(
        self,
        minimum_year: int = datetime.now().year - 100,
        maximum_year: int = datetime.now().year,
        allowed_pos_tags: set[str] | None = None,
    ):
        """
        Initialize the Wiktextract processor.

        Args:
            minimum_year (int): Minimum year for filtering example sentences.
            maximum_year (int): Maximum year for filtering example sentences.
            allowed_pos_tags (set[str] | None): Set of allowed part-of-speech tags. If None, all POS tags are allowed.
        """
        self._minimum_year: int = minimum_year
        self._maximum_year: int = maximum_year

        self.allowed_pos_tags: set[str] | None = allowed_pos_tags

        self._nlp: Language = spacy.load("en_core_web_sm", disable=["parser", "ner"])

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

            leading_whitespace: int = len(text) - len(text.lstrip())
            sentence: str = text.strip()

            bold_text_offsets: list[tuple[int, int]] = [
                (start - leading_whitespace, end - leading_whitespace)
                for start, end in example.get("bold_text_offsets", [])
            ]

            reference: str | None = example.get("ref")
            if reference:
                year: int | None = self._extract_year(reference)
                if year is None:
                    continue

                if self._minimum_year <= year <= self._maximum_year:
                    sentences.append(
                        Quotation(
                            sentence=sentence,
                            word_offsets=bold_text_offsets,
                            reference=reference.strip(),
                        )
                    )
            elif example.get("type", "") == "example":
                sentences.append(
                    Example(
                        sentence=sentence,
                        word_offsets=bold_text_offsets,
                    )
                )

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
    ) -> dict[str, dict[str, list[str]]]:
        """
        Extract translations from a Wiktextract entry.

        Args:
            raw_translations (list[dict[str, Any]]): List of raw translations.

        Returns:
            dict[str, dict[str, list[str]]]: A nested dictionary mapping sense definitions to languages and their corresponding translations.
        """
        translations: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )

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

            if word not in translations[sense][language]:
                translations[sense][language].append(word)

        return translations

    def _merge_records(
        self,
        existing_record: dict[str, Any],
        record: dict[str, Any],
    ) -> None:
        """
        Merge two lemma records, combining their senses and translations while avoiding duplicates.

        Args:
            existing_record (dict[str, Any]): The existing lemma record to merge into.
            record (dict[str, Any]): The new lemma record to merge from.
        """
        for i, sense in enumerate(record["senses"]):
            existing_sense: Sense = existing_record["senses"][i]

            existing_sentences: set[str] = {
                sentence.sentence for sentence in existing_sense.sentences
            }

            for sentence in sense.sentences:
                if sentence.sentence not in existing_sentences:
                    existing_sense.sentences.append(sentence)
                    existing_sentences.add(sentence.sentence)

        for definition, translation_map in record["translations"].items():
            if definition not in existing_record["translations"]:
                existing_record["translations"][definition] = {
                    language: translations[:]
                    for language, translations in translation_map.items()
                }
            else:
                for language, translations in translation_map.items():
                    if language not in existing_record["translations"][definition]:
                        existing_record["translations"][definition][language] = (
                            translations[:]
                        )
                    else:
                        for translation in translations:
                            if (
                                translation
                                not in existing_record["translations"][definition][
                                    language
                                ]
                            ):
                                existing_record["translations"][definition][
                                    language
                                ].append(translation)

    def _find_word_offsets(
        self,
        sentence_doc: Doc,
        lemma_doc: Doc,
        bold_text_offsets: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """
        Find offsets of the lemma in the sentence, including bold text offsets.

        Args:
            sentence_doc (Doc): The spaCy Doc object for the sentence.
            lemma_doc (Doc): The spaCy Doc object for the lemma.
            bold_text_offsets (list[tuple[int, int]]): List of offsets for bold text

        Returns:
            list[tuple[int, int]]: List of offsets where the lemma is found in the
        """
        word_offsets: list[tuple[int, int]] = []

        lemma_texts: list[str] = [token.text.lower() for token in lemma_doc]
        lemma_lemmas: list[str] = [token.lemma_.lower() for token in lemma_doc]

        for i in range(len(sentence_doc) - len(lemma_doc) + 1):
            window: Span = sentence_doc[i : i + len(lemma_doc)]

            if [token.text.lower() for token in window] == lemma_texts or [
                token.lemma_.lower() for token in window
            ] == lemma_lemmas:
                start: int = window[0].idx
                end: int = window[-1].idx + len(window[-1].text)

                if (start, end) not in word_offsets:
                    word_offsets.append((start, end))

        for start, end in bold_text_offsets:
            tokens: list[Token] = [
                token
                for token in sentence_doc
                if token.idx >= start and token.idx + len(token.text) <= end
            ]

            if [token.text.lower() for token in tokens] == lemma_texts or [
                token.lemma_.lower() for token in tokens
            ] == lemma_lemmas:
                if (start, end) not in word_offsets:
                    word_offsets.append((start, end))

        return word_offsets

    def extract_lemmas(
        self,
        input_path: Path,
        batch_size: int = 1000,
        n_process: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Extract lemmas and their translations from the Wiktextract JSONL file.

        Args:
            input_path (Path): Path to the Wiktextract JSONL file.
            batch_size (int): Number of lines to process in each batch for finding word offsets.
            n_process (int): Number of processes to use for finding word offsets. If 1, processing will be done sequentially.

        Returns:
            list[dict[str, Any]]: List of dictionaries containing lemma information.
        """
        records: dict[str, dict[str, Any]] = {}

        with (
            gzip.open(input_path, "rt", encoding="utf-8") as file,
            tqdm(
                desc="Extracting",
                unit=" lines",
            ) as pbar,
        ):
            for line in file:
                lemma_entry: dict[str, Any] = json.loads(line)

                language: str | None = lemma_entry.get("lang_code") or lemma_entry.get("lang")
                if language and language.lower() in ("en", "english"):
                    lemma: str | None = lemma_entry.get("word")
                    if lemma:
                        lemma = lemma.strip()

                        pos_tag: str | None = lemma_entry.get("pos")
                        if (
                            self.allowed_pos_tags is None
                            or pos_tag in self.allowed_pos_tags
                        ):
                            senses: list[Sense] = self._extract_senses(
                                lemma_entry.get("senses", []),
                            )

                            if senses:
                                record_id: str = str(
                                    uuid.uuid5(
                                        uuid.NAMESPACE_DNS,
                                        f"{lemma}|{pos_tag}|{'|'.join(sense.definition.lower() for sense in senses)}",
                                    )
                                )

                                translations: dict[str, dict[str, list[str]]] = (
                                    self._extract_translations(
                                        lemma_entry.get("translations", []),
                                    )
                                )

                                record: dict[str, Any] = {
                                    "lemma": lemma,
                                    "pos": pos_tag,
                                    "senses": senses,
                                    "translations": translations,
                                }

                                if record_id in records:
                                    self._merge_records(records[record_id], record)
                                else:
                                    records[record_id] = record

                pbar.update(1)

        lemmas: list[dict[str, Any]] = [
            {"id": record_id, **record} for record_id, record in records.items()
        ]

        sentences: list[tuple[str, Sentence]] = [
            (lemma["lemma"].lower(), sentence)
            for lemma in lemmas
            for sense in lemma["senses"]
            for sentence in sense.sentences
        ]

        for sentence_doc, lemma_doc, (_, sentence) in tqdm(
            zip(
                self._nlp.pipe(
                    (sentence.sentence for _, sentence in sentences),
                    batch_size=batch_size,
                    n_process=n_process,
                ),
                self._nlp.pipe(
                    (lemma for lemma, _ in sentences),
                    batch_size=batch_size,
                    n_process=n_process,
                ),
                sentences,
            ),
            desc="Finding word offsets",
            total=len(sentences),
            unit=" lemma",
        ):
            sentence.word_offsets = self._find_word_offsets(
                sentence_doc,
                lemma_doc,
                sentence.word_offsets,
            )

        return lemmas

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
    ) -> dict[str, dict[str, str]]:
        """
        Build a mapping dictionary from the mappings JSONL file.

        Args:
            mappings_path (Path): Path to the mappings JSONL file.

        Returns:
            dict[str, dict[str, str]]: A dictionary mapping lemma IDs to their corresponding sense-to-translation mappings.
        """
        mappings: dict[str, dict[str, str]] = {}

        with mappings_path.open(encoding="utf-8") as file:
            for line in file:
                entry = self._safe_load(line)
                if not entry:
                    continue

                entry_id: str = entry.get("id", "")

                if entry_id in mappings:
                    mappings[entry_id] = {}
                    continue

                raw_mapping: dict[str, str] | None = entry.get("mapping")
                if isinstance(raw_mapping, dict):
                    mappings.setdefault(entry_id, raw_mapping)

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
        mappings: dict[str, dict[str, str]] = self._build_mappings(
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

                record_id: str = input_entry.get("id", "")

                lemma: str = input_entry.get("lemma", "")
                pos: str = input_entry.get("pos", "")

                translation_map: dict[str, dict[str, list[str]]] = input_entry.get(
                    "translations", {}
                )

                translation_keys: list[str] = [
                    key for key in translation_map.keys() if isinstance(key, str)
                ]

                mapping: dict[str, str] = (
                    mappings.get(record_id, {}) if mappings else {}
                )

                senses: list[dict[str, Any]] = []

                for i, sense in enumerate(input_entry.get("senses", []), start=1):
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

                    sense_entry: dict[str, Any] = {
                        "sense_order": sense.get("sense_order"),
                        "definition": sense.get("definition"),
                        "sentences": sense.get("sentences"),
                    }

                    if translations:
                        sense_entry["translations"] = translations

                    senses.append(sense_entry)

                yield {
                    "id": record_id,
                    "lemma": lemma,
                    "pos": pos,
                    "senses": senses,
                }

                pbar.update(1)
