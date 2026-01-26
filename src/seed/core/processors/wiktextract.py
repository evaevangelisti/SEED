import gzip
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from ...config import (
    DEFAULT_DEVICE,
    DEFAULT_EMBEDDER,
    DEFAULT_GAP,
    DEFAULT_MINIMUM_YEAR,
    DEFAULT_THRESHOLD,
)
from ...models import Lemma, Sense, Translation
from .base import Processor


class WiktextractProcessor(Processor):
    """
    Processor for Wiktextract JSONL files.
    """

    _YEAR_PATTERN: re.Pattern = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")

    def __init__(
        self,
        input_path: Path,
        minimum_year: int = DEFAULT_MINIMUM_YEAR,
        maximum_year: int = datetime.now().year,
        embedder: str = DEFAULT_EMBEDDER,
        device: str = DEFAULT_DEVICE,
        threshold: float = DEFAULT_THRESHOLD,
        gap: float = DEFAULT_GAP,
    ):
        """
        Initialize the Wiktextract processor.

        Args:
            input_path (Path): Path to the Wiktextract JSONL file.
            minimum_year (int): Minimum year for filtering example sentences.
            maximum_year (int): Maximum year for filtering example sentences.
            embedder (str): Embedder model name.
            device (str): Device for embedding model.
            threshold (float): Similarity threshold for associating translations.
            gap (float): Similarity gap for associating translations.
        """
        self._input_path: Path = input_path

        self._minimum_year: int = minimum_year
        self._maximum_year: int = maximum_year

        self._embedder: SentenceTransformer = SentenceTransformer(
            embedder,
            device=device,
        )

        self._threshold: float = threshold
        self._gap: float = gap

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
    ) -> list[str]:
        """
        Extract example sentences from Wiktextract examples.

        Args:
            raw_sentences (list[dict[str, Any]]): List of raw example sentences.

        Returns:
            list[str]: List of extracted example sentences.
        """
        sentences: list[str] = []

        for example in raw_sentences:
            text: str | None = example.get("text")
            if not text:
                continue

            reference: str | None = example.get("ref")

            if reference:
                year: int | None = self._extract_year(reference)
                if year is None:
                    continue

                if self._minimum_year <= year <= self._maximum_year:
                    sentences.append(text.strip())
            elif example.get("type", "") == "example":
                sentences.append(text.strip())

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
            list[Sense]: List of extracted senses.
        """
        senses: list[Sense] = []

        for i, sense in enumerate(raw_senses):
            glosses: list[str] = sense.get("glosses", [])
            if not glosses:
                continue

            definition: str = glosses[0].strip()
            if not definition:
                continue

            sentences: list[str] = self._extract_sentences(sense.get("examples", []))
            if not sentences:
                continue

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

            word: str | None = translation.get("word")
            if not word:
                continue

            language: str | None = translation.get("lang")
            if not language:
                continue

            translations[sense.strip()].append(
                Translation(
                    translation=word.strip().lower(),
                    language=language.strip().lower(),
                )
            )

        return translations

    def _associate_translations(
        self,
        senses: list[Sense],
        translations: dict[str, list[Translation]],
    ) -> None:
        """
        Associate translations with senses using semantic similarity.

        Args:
            senses (list[Sense]): List of Sense objects.
            translations (dict[str, list[Translation]]): Dictionary mapping sense definitions to lists of Translations.
        """
        if not senses or not translations:
            return

        sense_definitions: list[str] = [sense.definition for sense in senses]

        sense_embeddings: torch.Tensor = self._embedder.encode(
            sense_definitions,
            convert_to_tensor=True,
        )

        translation_definitions: list[str] = list(translations.keys())

        translation_embeddings = self._embedder.encode(
            translation_definitions,
            convert_to_tensor=True,
        )

        similarities: torch.Tensor = util.cos_sim(
            translation_embeddings,
            sense_embeddings,
        )

        candidates: list[tuple[float, int, str]] = []

        for i, translation_definition in enumerate(translation_definitions):
            scores: torch.Tensor = similarities[i]
            if scores.numel() == 0:
                continue

            top_scores: torch.Tensor
            top_indices: torch.Tensor

            top_scores, top_indices = torch.topk(scores, k=min(2, scores.numel()))

            best_index: int = int(top_indices[0])
            best_score: float = float(top_scores[0])

            if best_score < self._threshold:
                continue

            second_score: float = (
                float(top_scores[1]) if top_scores.numel() > 1 else 0.0
            )

            if best_score - second_score < self._gap:
                continue

            candidates.append((best_score, best_index, translation_definition))

        candidates.sort(key=lambda candidate: candidate[0], reverse=True)

        associated_senses: torch.Tensor = torch.zeros(len(senses), dtype=torch.bool)

        for best_score, best_index, translation_definition in candidates:
            if associated_senses[best_index]:
                continue

            senses[best_index].translations.extend(translations[translation_definition])
            associated_senses[best_index] = True

    def process(
        self,
    ) -> Generator[Lemma, None, None]:
        """
        Process the Wiktextract JSONL file and yield Lemma objects.

        Returns:
            Generator[Lemma, None, None]: Generator of Lemma objects.
        """
        lemmas: dict[str, list[Sense]] = {}

        with (
            gzip.open(self._input_path, "rt", encoding="utf-8") as file,
            tqdm(
                desc="Processing",
                unit=" lines",
            ) as pbar,
        ):
            for line in file:
                entry: dict[str, Any] = json.loads(line)

                language: str | None = entry.get("lang_code") or entry.get("lang")
                if language and language.lower() in ("en", "english"):
                    lemma: str | None = entry.get("word")
                    if lemma:
                        senses: list[Sense] = self._extract_senses(
                            entry.get("senses", []),
                        )

                        if senses:
                            self._associate_translations(
                                senses,
                                self._extract_translations(
                                    entry.get("translations", []),
                                ),
                            )

                            lemmas.setdefault(lemma.strip().lower(), []).extend(senses)

                pbar.update(1)

        for lemma, senses in lemmas.items():
            yield Lemma(lemma=lemma, senses=senses)
