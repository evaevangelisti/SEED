from pathlib import Path

# Defaults

DEFAULT_CHUNK_SIZE: int = 1024 * 1024
DEFAULT_TIMEOUT: int = 60

DEFAULT_BUFFER_SIZE: int = 1024 * 1024

# Paths

ROOT: Path = Path(__file__).resolve().parent.parent.parent

DATA_DIR: Path = ROOT / "data"

RAW_DIR: Path = DATA_DIR / "raw"

COMPRESSED_WIKTEXTRACT_PATH: Path = RAW_DIR / "wiktextract.jsonl.gz"
DEFAULT_MAPPINGS_PATH: Path = RAW_DIR / "mappings.jsonl"

INTERIM_DIR: Path = DATA_DIR / "interim"

WIKTEXTRACT_PATH: Path = INTERIM_DIR / "wiktextract.jsonl"

PROCESSED_DIR: Path = DATA_DIR / "processed"

SEED_PATH: Path = PROCESSED_DIR / "seed.jsonl"

# URLs

WIKTEXTRACT_URL: str = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"
