from typing import Any

import typer

from .models import POS


def parse_allowed_pos_tags(
    allowed_pos_tags: str | None,
) -> set[POS] | None:
    """
    Parse a comma-separated string of POS tags into a list of strings. If the input is None, return None.

    Args:
        pos_tags (str | None): A comma-separated string of POS tags, or None.

    Returns:
        set[POS] | None: A set of POS tags, or None if the input was None.
    """
    if allowed_pos_tags is None:
        return None

    pos_tags: dict[str, POS] = {pos_tag.value: pos_tag for pos_tag in POS}

    return {
        pos_tags[pos_tag.strip()]
        for pos_tag in allowed_pos_tags.split()
        if pos_tag.strip()
    }


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
    ),
    force_download: bool = typer.Option(
        False,
        help="Force re-download",
    ),
) -> None:
    parsed_allowed_pos_tags = parse_allowed_pos_tags(allowed_pos_tags)

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

    from .config import WIKTEXTRACT_LEMMAS_PATH
    from .exporters import ExporterFactory
    from .processors import WiktextractProcessor

    wiktextract_processor: WiktextractProcessor = WiktextractProcessor(
        COMPRESSED_WIKTEXTRACT_PATH,
        minimum_year=minimum_year,
        maximum_year=maximum_year,
        allowed_pos_tags=parsed_allowed_pos_tags,
    )

    exporter, _ = ExporterFactory.create(WIKTEXTRACT_LEMMAS_PATH)
    exporter.export(wiktextract_processor.extract_lemmas())

    typer.echo(f"Saved Wiktextract lemmas to {WIKTEXTRACT_LEMMAS_PATH}")

    from .config import WIKTEXTRACT_TRANSLATIONS_PATH

    exporter, _ = ExporterFactory.create(WIKTEXTRACT_TRANSLATIONS_PATH)
    exporter.export(wiktextract_processor.extract_translations())

    typer.echo(f"Saved Wiktextract translations to {WIKTEXTRACT_TRANSLATIONS_PATH}")

    from .config import WORDNET_PATH
    from .processors import WordNetProcessor

    wordnet_processor: WordNetProcessor = WordNetProcessor(
        allowed_pos_tags=parsed_allowed_pos_tags,
    )

    exporter, _ = ExporterFactory.create(WORDNET_PATH)
    exporter.export(wordnet_processor.extract_lemmas())

    typer.echo(f"Saved WordNet lemmas to {WORDNET_PATH}")

    from .config import TRANSLATION_MAPPINGS_PATH, WIKTIONARY_PATH
    from .mapper import Mapper

    mapper: Mapper = Mapper(WIKTEXTRACT_LEMMAS_PATH)

    lemmas: list[dict[str, Any]] = []

    if TRANSLATION_MAPPINGS_PATH.exists():
        lemmas = mapper.associate_translations(
            TRANSLATION_MAPPINGS_PATH,
        )
    else:
        typer.echo(
            f"Mappings file not found at {TRANSLATION_MAPPINGS_PATH}, skipping association of translations"
        )

    from .config import WORDNET_SYNSET_IDS_MAPPINGS_PATH

    if WORDNET_SYNSET_IDS_MAPPINGS_PATH.exists():
        lemmas = mapper.associate_wordnet_synset_ids(
            WORDNET_SYNSET_IDS_MAPPINGS_PATH,
        )
    else:
        typer.echo(
            f"Mappings file not found at {WORDNET_SYNSET_IDS_MAPPINGS_PATH}, skipping association of WordNet definitions"
        )

    if lemmas:
        exporter, _ = ExporterFactory.create(WIKTIONARY_PATH)
        exporter.export((lemma for lemma in lemmas if lemma["sense"]["sentences"]))

        typer.echo(f"Saved Wiktionary data to {WIKTIONARY_PATH}")


if __name__ == "__main__":
    app()
