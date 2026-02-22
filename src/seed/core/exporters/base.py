from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Mapping

from ...config import DEFAULT_BUFFER_SIZE


class Exporter(ABC):
    """
    Base class for exporters.
    """

    def __init__(
        self,
        output_path: Path,
    ):
        """
        Initialize the exporter.

        Args:
            output_path (Path): Output path file.
        """
        self._output_path = output_path
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def export(
        self,
        data: Iterable[Mapping[str, Any] | object],
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> Path:
        """
        Export lemmas to the specified output path.

        Args:
            data (Iterable[Mapping[str, Any] | object]): Data to be exported.
            buffer_size (int): Buffer size for file writing.

        Returns:
            Path: Output path file.
        """
        pass
