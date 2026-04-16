import typer


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
    minimum_year: int | None = typer.Option(
        None,
        help="Minimum year",
    ),
    maximum_year: int | None = typer.Option(
        None,
        help="Maximum year",
    ),
    allowed_pos_tags: str | None = typer.Option(
        None,
        help="Allowed POS tags",
        callback=parse_allowed_pos_tags,
    ),
    force_download: bool = typer.Option(
        False,
        help="Force re-download",
    ),
) -> None:
    from .config import COMPRESSED_WIKTEXTRACT_PATH, WIKTEXTRACT_URL
    from .downloader import Downloader

    downloader: Downloader = Downloader(
        WIKTEXTRACT_URL,
        COMPRESSED_WIKTEXTRACT_PATH,
    )

    if force_download or not COMPRESSED_WIKTEXTRACT_PATH.exists():
        downloader.download()

        typer.echo(f"Downloaded Wiktextract data to {COMPRESSED_WIKTEXTRACT_PATH}")
    else:
        typer.echo(f"Wiktextract data already exists at {COMPRESSED_WIKTEXTRACT_PATH}")

    from .config import WIKTEXTRACT_PATH
    from .exporters import ExporterFactory
    from .processors import WiktextractProcessor

    processor: WiktextractProcessor = WiktextractProcessor(
        minimum_year=minimum_year,
        maximum_year=maximum_year,
        allowed_pos_tags=(
            set(allowed_pos_tags) if allowed_pos_tags is not None else None
        ),
    )

    exporter, _ = ExporterFactory.create(WIKTEXTRACT_PATH)
    exporter.export(
        processor.extract_lemmas(
            COMPRESSED_WIKTEXTRACT_PATH,
        ),
    )

    typer.echo(f"Saved processed data to {WIKTEXTRACT_PATH}")

    from .config import DEFAULT_MAPPINGS_PATH, WIKTIONARY_PATH

    if DEFAULT_MAPPINGS_PATH.exists():
        exporter, _ = ExporterFactory.create(WIKTIONARY_PATH)
        exporter.export(
            processor.associate_translations(WIKTEXTRACT_PATH, DEFAULT_MAPPINGS_PATH),
        )

        typer.echo(f"Saved Wiktionary data to {WIKTIONARY_PATH}")
    else:
        typer.echo(
            f"Mappings file not found at {DEFAULT_MAPPINGS_PATH}, skipping association of translations"
        )


if __name__ == "__main__":
    app()
