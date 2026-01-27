from datetime import datetime
from pathlib import Path

import typer

from .config import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_EMBEDDER,
    DEFAULT_MINIMUM_YEAR,
    DEFAULT_TIMEOUT,
    ROOT,
    SEED_PATH,
    WIKTEXTRACT_PATH,
    WIKTEXTRACT_URL,
)

app: typer.Typer = typer.Typer(add_completion=False)


@app.command()
def main(
    force_download: bool = typer.Option(
        False,
        help="Force re-download",
    ),
    chunk_size: int = typer.Option(
        DEFAULT_CHUNK_SIZE,
        help="Chunk size in bytes",
    ),
    timeout: int = typer.Option(
        DEFAULT_TIMEOUT,
        help="Request timeout in seconds",
    ),
    embedder: str = typer.Option(
        DEFAULT_EMBEDDER,
        help="Embedder model name",
    ),
    device: str = typer.Option(
        DEFAULT_DEVICE,
        help="Device for embedding model",
    ),
    minimum_year: int = typer.Option(
        DEFAULT_MINIMUM_YEAR,
        help="Minimum year",
    ),
    maximum_year: int = typer.Option(
        datetime.now().year,
        help="Maximum year",
    ),
    output_path: Path = typer.Option(
        SEED_PATH,
        show_default=SEED_PATH.relative_to(ROOT.parent).as_posix(),
        help="Output path",
    ),
    buffer_size: int = typer.Option(
        DEFAULT_BUFFER_SIZE,
        help="Buffer size in bytes",
    ),
) -> None:
    from dotenv import load_dotenv

    load_dotenv()

    from .core import Downloader

    downloader: Downloader = Downloader(
        WIKTEXTRACT_URL,
        WIKTEXTRACT_PATH,
    )

    downloaded: bool = downloader.download(
        force_download=force_download,
        chunk_size=chunk_size,
        timeout=timeout,
    )

    if downloaded:
        typer.echo(f"Downloaded Wiktextract data to {WIKTEXTRACT_PATH}")
    else:
        typer.echo(f"Wiktextract data already exists at {WIKTEXTRACT_PATH}")

    from .core import ExporterFactory, WiktextractProcessor
    from .core.exporters.base import Exporter

    processor: WiktextractProcessor = WiktextractProcessor(
        WIKTEXTRACT_PATH,
        embedder=embedder,
        device=device,
        minimum_year=minimum_year,
        maximum_year=maximum_year,
    )

    exporter: Exporter
    exporter, output_path = ExporterFactory.create(output_path)

    exporter.export(
        processor.process(),
        buffer_size=buffer_size,
    )

    typer.echo(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    app()
