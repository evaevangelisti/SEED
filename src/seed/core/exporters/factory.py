from pathlib import Path
from typing import Any, Callable, Type

from .base import Exporter

DEFAULT_EXPORTER: str = "jsonl"


class ExporterFactory:
    """
    Factory class for creating exporters based on file extensions.
    """

    _registry: dict[str, Type[Exporter]] = {}

    @classmethod
    def register(
        cls,
        *extensions: str,
    ) -> Callable[[Type[Exporter]], Type[Exporter]]:
        """
        Registers an exporter for the given file extensions.

        Args:
            *extensions (str): File extensions.

        Returns:
            Callable[[Type[Exporter]], Type[Exporter]]: Decorator function.
        """

        def decorator(
            exporter: Type[Exporter],
        ) -> Type[Exporter]:
            """
            Registers the exporter.

            Args:
                exporter (Type[Exporter]): Exporter class.

            Returns:
                Type[Exporter]: Registered exporter class.
            """
            for extension in extensions:
                cls._registry[extension.lower()] = exporter

            return exporter

        return decorator

    @classmethod
    def create(
        cls,
        output_path: Path,
        **kwargs: Any,
    ) -> tuple[Exporter, Path]:
        """
        Creates an exporter based on the file extension.

        Args:
            output_path (Path): Output file path.
            **kwargs (Any): Additional keyword arguments for the exporter.

        Returns:
            tuple[Exporter, Path]: Exporter instance and output path.
        """
        extension: str = output_path.suffix.lstrip(".").lower()

        exporter: Type[Exporter] | None = cls._registry.get(extension)
        if exporter is None:
            exporter = cls._registry.get(DEFAULT_EXPORTER)
            if exporter is None:
                raise ValueError("error: No default exporter registered")

            output_path = output_path.with_suffix(f".{DEFAULT_EXPORTER}")

        return exporter(output_path=output_path, **kwargs), output_path
