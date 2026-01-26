from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator

from ...config import DEFAULT_BUFFER_SIZE
from ...models import Lemma


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
        lemmas: Generator[Lemma, None, None],
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> Path:
        """
        Export lemmas to the specified output path.

        Args:
            lemmas (Generator[Lemma, None, None]): Generator of Lemma object.
            buffer_size (int): Buffer size for file writing.

        Returns:
            Path: Output path file.
        """
        pass
