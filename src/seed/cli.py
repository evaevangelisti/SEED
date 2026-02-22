from datetime import datetime
from pathlib import Path

import typer

from .config import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAPPINGS_PATH,
    DEFAULT_TIMEOUT,
    ROOT,
    SEED_PATH,
)

app: typer.Typer = typer.Typer(add_completion=False)


@app.command()
def main(
    force: bool = typer.Option(
        False,
        help="Force re-download and re-processing of data",
    ),
    chunk_size: int = typer.Option(
        DEFAULT_CHUNK_SIZE,
        help="Chunk size in bytes",
    ),
    timeout: int = typer.Option(
        DEFAULT_TIMEOUT,
        help="Request timeout in seconds",
    ),
    minimum_year: int = typer.Option(
        datetime.now().year - 25,
        help="Minimum year",
    ),
    maximum_year: int = typer.Option(
        datetime.now().year,
        help="Maximum year",
    ),
    buffer_size: int = typer.Option(
        DEFAULT_BUFFER_SIZE,
        help="Buffer size in bytes",
    ),
    mappings_path: Path = typer.Option(
        DEFAULT_MAPPINGS_PATH,
        help="Mapping path",
    ),
    output_path: Path = typer.Option(
        SEED_PATH,
        show_default=SEED_PATH.relative_to(ROOT.parent).as_posix(),
        help="Output directory",
    ),
) -> None:
    from .config import COMPRESSED_WIKTEXTRACT_PATH, WIKTEXTRACT_URL
    from .core import Downloader

    downloader: Downloader = Downloader(
        WIKTEXTRACT_URL,
        COMPRESSED_WIKTEXTRACT_PATH,
    )

    downloaded: bool = downloader.download(
        force_download=force,
        chunk_size=chunk_size,
        timeout=timeout,
    )

    if downloaded:
        typer.echo(f"Downloaded Wiktextract data to {COMPRESSED_WIKTEXTRACT_PATH}")
    else:
        typer.echo(f"Wiktextract data already exists at {COMPRESSED_WIKTEXTRACT_PATH}")

    from .config import WIKTEXTRACT_PATH
    from .core import ExporterFactory, WiktextractProcessor

    processor: WiktextractProcessor = WiktextractProcessor(
        minimum_year=minimum_year,
        maximum_year=maximum_year,
    )

    if force or not WIKTEXTRACT_PATH.exists():
        exporter, _ = ExporterFactory.create(WIKTEXTRACT_PATH)
        exporter.export(
            processor.extract_lemmas(WIKTEXTRACT_PATH),
            buffer_size=buffer_size,
        )

        typer.echo(f"Saved processed data to {WIKTEXTRACT_PATH}")
    else:
        typer.echo(f"Processed data already exists at {WIKTEXTRACT_PATH}")

    if mappings_path.exists():
        exporter, _ = ExporterFactory.create(SEED_PATH)
        exporter.export(
            processor.associate_translations(WIKTEXTRACT_PATH, mappings_path),
            buffer_size=buffer_size,
        )

        typer.echo(f"Saved SEED data to {SEED_PATH}")
    else:
        typer.echo(
            f"Mappings file not found at {mappings_path}, skipping association of translations"
        )


if __name__ == "__main__":
    app()
