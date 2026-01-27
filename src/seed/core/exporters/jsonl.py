import json
from dataclasses import asdict
from pathlib import Path
from typing import Generator

from tqdm import tqdm

from ...config import DEFAULT_BUFFER_SIZE
from ...models import Lemma
from .base import Exporter
from .factory import ExporterFactory


@ExporterFactory.register("jsonl")
class JsonlExporter(Exporter):
    """
    Exporter for JSONL format.
    """

    def export(
        self,
        lemmas: Generator[Lemma, None, None],
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> Path:
        """
        Export lemmas to a JSONL file.

        Args:
            lemmas (Generator[Lemma, None, None]): Generator of Lemma object.
            buffer_size (int): Buffer size for file writing.

        Returns:
            Path: Output path file.
        """
        with self._output_path.open(
            "w",
            encoding="utf-8",
            buffering=buffer_size,
        ) as file:
            for lemma in tqdm(lemmas, desc="Exporting", unit=" lemma"):
                file.write(f"{json.dumps(asdict(lemma), ensure_ascii=False)}\n")

        return self._output_path
