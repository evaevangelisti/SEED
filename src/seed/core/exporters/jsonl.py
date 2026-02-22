import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from tqdm import tqdm

from ...config import DEFAULT_BUFFER_SIZE
from .base import Exporter
from .factory import ExporterFactory


@ExporterFactory.register("jsonl")
class JsonlExporter(Exporter):
    """
    Exporter for JSONL format.
    """

    @classmethod
    def _serialize(
        cls,
        object: Any,
    ) -> Any:
        """
        Serialize an object to a JSON-serializable format.

        Args:
            object: The object to serialize.

        Returns:
            A JSON-serializable representation of the object.
        """
        match object:
            case _ if is_dataclass(object) and not isinstance(object, type):
                return {k: cls._serialize(v) for k, v in asdict(object).items()}

            case list():
                return [cls._serialize(item) for item in object]

            case dict():
                return {k: cls._serialize(v) for k, v in object.items()}

            case _:
                return object

    def export(
        self,
        data: Iterable[Mapping[str, Any] | object],
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> Path:
        """
        Export lemmas to a JSONL file.

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
            for item in tqdm(data, desc="Exporting", unit=" item"):
                file.write(f"{json.dumps(self._serialize(item), ensure_ascii=False)}\n")

        return self._output_path
