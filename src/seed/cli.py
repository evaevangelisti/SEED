from datetime import datetime

import typer

from .config import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAPPINGS_PATH,
    DEFAULT_TIMEOUT,
)


def parse_allowed_pos_tags(
    allowed_pos_tags: str | None,
) -> list[str] | None:
    """
    Parse a comma-separated string of POS tags into a list of strings. If the input is None, return None.

    Args:
        pos_tags (str | None): A comma-separated string of POS tags, or None.

    Returns:
        list[str] | None: A list of POS tags, or None if the input was None.
    """
    if allowed_pos_tags is None:
        return None

    return allowed_pos_tags.split()


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
        datetime.now().year - 100,
        help="Minimum year",
    ),
    maximum_year: int = typer.Option(
        datetime.now().year,
        help="Maximum year",
    ),
    allowed_pos_tags: str | None = typer.Option(
        None,
        help="Allowed POS tags",
        callback=parse_allowed_pos_tags,
    ),
    batch_size: int = typer.Option(
        1000,
        help="Batch size for processing data",
    ),
    n_process: int = typer.Option(
        1,
        help="Number of processes to use for processing data",
    ),
    buffer_size: int = typer.Option(
        DEFAULT_BUFFER_SIZE,
        help="Buffer size in bytes",
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

    from .config import SEED_PATH, WIKTEXTRACT_PATH
    from .core import ExporterFactory, WiktextractProcessor

    processor: WiktextractProcessor = WiktextractProcessor(
        minimum_year=minimum_year,
        maximum_year=maximum_year,
        allowed_pos_tags=(
            set(allowed_pos_tags) if allowed_pos_tags is not None else None
        ),
    )

    if force or not WIKTEXTRACT_PATH.exists():
        exporter, _ = ExporterFactory.create(WIKTEXTRACT_PATH)
        exporter.export(
            processor.extract_lemmas(
                COMPRESSED_WIKTEXTRACT_PATH,
                batch_size=batch_size,
                n_process=n_process,
            ),
            buffer_size=buffer_size,
        )

        typer.echo(f"Saved processed data to {WIKTEXTRACT_PATH}")
    else:
        typer.echo(f"Processed data already exists at {WIKTEXTRACT_PATH}")

    if DEFAULT_MAPPINGS_PATH.exists():
        exporter, _ = ExporterFactory.create(SEED_PATH)
        exporter.export(
            processor.associate_translations(WIKTEXTRACT_PATH, DEFAULT_MAPPINGS_PATH),
            buffer_size=buffer_size,
        )

        typer.echo(f"Saved SEED data to {SEED_PATH}")
    else:
        typer.echo(
            f"Mappings file not found at {DEFAULT_MAPPINGS_PATH}, skipping association of translations"
        )


if __name__ == "__main__":
    app()
