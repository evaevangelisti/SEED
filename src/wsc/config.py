from pathlib import Path

# Defaults

DEFAULT_CHUNK_SIZE: int = 1024 * 1024
DEFAULT_TIMEOUT: int = 60

DEFAULT_BATCH_SIZE: int = 1024
DEFAULT_N_PROCESS: int = 4

DEFAULT_BUFFER_SIZE: int = 1024 * 1024

# External Resources

NLTK_RESOURCES: list[str] = ["wordnet", "omw-1.4"]

# Paths

ROOT: Path = Path(__file__).resolve().parent.parent.parent

DATA_DIR: Path = ROOT / "data"

RAW_DIR: Path = DATA_DIR / "raw"

COMPRESSED_WIKTEXTRACT_PATH: Path = RAW_DIR / "wiktextract.jsonl.gz"

TRANSLATION_MAPPINGS_PATH: Path = RAW_DIR / "translation_mappings.json"
WORDNET_SYNSET_IDS_MAPPINGS_PATH: Path = RAW_DIR / "wordnet_synset_ids_mappings.json"

INTERIM_DIR: Path = DATA_DIR / "interim"

WIKTEXTRACT_LEMMAS_PATH: Path = INTERIM_DIR / "wiktextract_lemmas.jsonl"
WIKTEXTRACT_TRANSLATIONS_PATH: Path = INTERIM_DIR / "wiktextract_translations.jsonl"
WORDNET_PATH: Path = INTERIM_DIR / "wordnet.jsonl"

PROCESSED_DIR: Path = DATA_DIR / "processed"

WIKTIONARY_PATH: Path = PROCESSED_DIR / "wiktionary.jsonl"

# URLs

WIKTEXTRACT_URL: str = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"
