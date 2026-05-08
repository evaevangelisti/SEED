import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..config import DEFAULT_BUFFER_SIZE
from .base import Exporter
from .factory import ExporterFactory


@ExporterFactory.register("json")
class JsonExporter(Exporter):
    """
    Exporter for JSON format.
    """

    def export(
        self,
        data: Iterable[Mapping[str, Any] | object],
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> Path:
        """
        Export lemmas to a JSON file.

        Args:
            data (Iterable[Mapping[str, Any] | object]): Data to be exported.
            buffer_size (int): Buffer size for file writing.

        Returns:
            Path: Output path file.
        """
        with self._output_path.open(
            "w",
            encoding="utf-8",
            buffering=buffer_size,
        ) as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        return self._output_path
