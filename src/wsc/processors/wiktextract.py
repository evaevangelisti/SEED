import gzip
import json
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token
from tqdm import tqdm

from ..config import DEFAULT_BATCH_SIZE, DEFAULT_N_PROCESS
from ..models import POS, Example, Quotation, Sentence, WiktionaryLemma, WiktionarySense
from .base import Processor


class WiktextractProcessor(Processor):
    """
    Processor for Wiktextract JSONL files.
    """

    _YEAR_PATTERN: re.Pattern = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")

    def __init__(
        self,
        input_path: Path,
        minimum_year: int | None = None,
        maximum_year: int | None = None,
        allowed_pos_tags: set[POS] | None = None,
    ):
        """
        Initialize the Wiktextract processor.

        Args:
            input_path (Path): Path to the Wiktextract JSONL file containing lemmas and senses.
            minimum_year (int | None): Minimum year for filtering example sentences. If None, no minimum year filter is applied.
            maximum_year (int | None): Maximum year for filtering example sentences. If None, no maximum year filter is applied.
            allowed_pos_tags (set[POS] | None): Set of allowed part-of-speech tags. If None, all POS tags are allowed.
        """
        self._input_path: Path = input_path

        self._minimum_year: int | None = minimum_year
        self._maximum_year: int | None = maximum_year

        self._allowed_pos_tags: set[POS] | None = allowed_pos_tags

        self._records: dict[str, dict[str, Any]] | None = None
        self._nlp: Language = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    @classmethod
    def _extract_year(
        cls,
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

        year_match: re.Match | None = cls._YEAR_PATTERN.search(string)
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

                if self._minimum_year is not None and year < self._minimum_year:
                    continue

                if self._maximum_year is not None and year > self._maximum_year:
                    continue

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
    ) -> list[WiktionarySense]:
        """
        Extract senses from a Wiktextract entry.

        Args:
            raw_senses (list[dict[str, Any]]): List of raw senses.

        Returns:
            list[Sense]: List of extracted Sense objects.
        """
        processed_senses = []

        for raw_sense in raw_senses:
            glosses: list[str] = [
                gloss.strip()
                for gloss in raw_sense.get("raw_glosses")
                or raw_sense.get("glosses")
                or []
                if gloss.strip()
            ]

            if glosses:
                processed_senses.append((raw_sense, glosses))

        senses: list[WiktionarySense] = []

        for i, (raw_sense, current_glosses) in enumerate(processed_senses):
            if any(
                other_glosses[: len(current_glosses)] == current_glosses
                and len(other_glosses) > len(current_glosses)
                for j, (_, other_glosses) in enumerate(processed_senses)
                if i != j
            ):
                continue

            senses.append(
                WiktionarySense(
                    definition=current_glosses[-1],
                    sentences=self._extract_sentences(raw_sense.get("examples", [])),
                    parent_glosses=current_glosses[:-1] or None,
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
            existing_sense: WiktionarySense = existing_record["senses"][i]

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

    def _extract_records(
        self,
    ) -> dict[str, dict[str, Any]]:
        """
        Extract lemmas, their parts of speech, senses, and translations from the Wiktextract JSONL file.

        Returns:
            dict[str, dict[str, Any]]: A dictionary mapping unique record IDs to their corresponding lemma, part of speech, senses, and translations.
        """
        if self._records is not None:
            return self._records

        self._records = {}

        with (
            gzip.open(self._input_path, "rt", encoding="utf-8") as file,
            tqdm(
                desc="Extracting records from Wiktextract",
                unit=" lines",
            ) as pbar,
        ):
            for line in file:
                lemma_entry: dict[str, Any] = json.loads(line)

                language: str | None = lemma_entry.get(
                    "lang_code",
                ) or lemma_entry.get(
                    "lang",
                )

                if language and language.lower() in ("en", "english"):
                    lemma: str | None = lemma_entry.get("word")

                    if lemma:
                        lemma = lemma.strip()

                        pos_tag: POS | None = POS.from_wiktionary(
                            lemma_entry.get(
                                "pos",
                                "",
                            ),
                        )

                        if pos_tag is not None and (
                            self._allowed_pos_tags is None
                            or pos_tag in self._allowed_pos_tags
                        ):
                            senses: list[WiktionarySense] = self._extract_senses(
                                lemma_entry.get(
                                    "senses",
                                    [],
                                ),
                            )

                            if senses:
                                record_id: str = str(
                                    uuid.uuid5(
                                        uuid.NAMESPACE_DNS,
                                        f"{lemma}|{pos_tag.value}|{'|'.join(sense.definition.lower() for sense in senses)}",
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

                                if record_id in self._records:
                                    self._merge_records(
                                        self._records[record_id],
                                        record,
                                    )
                                else:
                                    self._records[record_id] = record

                pbar.update(1)

        return self._records

    @staticmethod
    def _find_word_offsets(
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

        lemma_texts: list[str] = [
            token.text.lower() for token in lemma_doc if token.text != "-"
        ]

        lemma_lemmas: list[str] = [
            token.lemma_.lower() for token in lemma_doc if token.lemma_ != "-"
        ]

        for i in range(len(sentence_doc)):
            collected_tokens: list[Token] = []
            j: int = i

            while j < len(sentence_doc) and len(collected_tokens) < len(lemma_texts):
                if sentence_doc[j].text != "-":
                    collected_tokens.append(sentence_doc[j])

                j += 1

            if len(collected_tokens) < len(lemma_texts):
                break

            if [token.text.lower() for token in collected_tokens] == lemma_texts or [
                token.lemma_.lower() for token in collected_tokens
            ] == lemma_lemmas:
                start: int = sentence_doc[i].idx
                end: int = sentence_doc[j - 1].idx + len(sentence_doc[j - 1].text)

                if (start, end) not in word_offsets:
                    word_offsets.append((start, end))

        for start, end in bold_text_offsets:
            tokens: list[Token] = [
                token
                for token in sentence_doc
                if token.idx >= start and token.idx + len(token.text) <= end
            ]

            if [
                token.text.lower() for token in tokens if token.text != "-"
            ] == lemma_texts or [
                token.lemma_.lower() for token in tokens if token.lemma_ != "-"
            ] == lemma_lemmas:
                if (start, end) not in word_offsets:
                    word_offsets.append((start, end))

        return word_offsets

    def extract_lemmas(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        n_process: int = DEFAULT_N_PROCESS,
    ) -> list[WiktionaryLemma]:
        """
        Extract lemmas, their parts of speech, senses, and translations from the Wiktextract JSONL file, and find offsets of the lemma in the example sentences.

        Args:
            batch_size (int): The batch size to use for processing sentences with spaCy. Defaults to DEFAULT_BATCH_SIZE.
            n_process (int): The number of processes to use for parallel processing with spaCy. Defaults to DEFAULT_N_PROCESS.

        Returns:
            list[WiktionaryLemma]: A list of WiktionaryLemma objects, each containing a lemma, its part of speech, senses with example sentences, and translations.
        """
        lemmas: list[WiktionaryLemma] = [
            WiktionaryLemma(
                id=record_id,
                lemma=record["lemma"],
                pos=record["pos"],
                senses=record["senses"],
            )
            for record_id, record in self._extract_records().items()
        ]

        sentences: list[Sentence] = [
            sentence
            for lemma in lemmas
            for sense in lemma.senses
            for sentence in sense.sentences
        ]

        docs: list[Doc] = list(
            tqdm(
                self._nlp.pipe(
                    [sentence.sentence for sentence in sentences]
                    + [
                        lemma.lemma.lower()
                        for lemma in lemmas
                        for sense in lemma.senses
                        for _ in sense.sentences
                    ],
                    batch_size=batch_size,
                    n_process=n_process,
                ),
                desc="Processing with spaCy",
                total=len(sentences) * 2,
                unit=" doc",
            )
        )

        for sentence_doc, lemma_doc, sentence in tqdm(
            zip(docs[: len(sentences)], docs[len(sentences) :], sentences),
            desc="Finding word offsets",
            total=len(sentences),
            unit=" sentence",
        ):
            sentence.word_offsets = self._find_word_offsets(
                sentence_doc,
                lemma_doc,
                sentence.word_offsets,
            )

        return lemmas

    def extract_translations(
        self,
    ) -> list[dict[str, Any]]:
        """
        Extract lemmas, their parts of speech, and their translations from the Wiktextract JSONL file.

        Returns:
            list[dict[str, Any]]: A list of dictionaries, each containing a lemma, its part of speech, and a list of its translations (mapped by sense definitions and languages).
        """
        return [
            {
                "id": record_id,
                "translations": record["translations"],
            }
            for record_id, record in tqdm(
                self._extract_records().items(),
                desc="Extracting translations from Wiktextract",
                unit=" record",
            )
        ]
