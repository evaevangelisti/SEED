from pathlib import Path

# Defaults

DEFAULT_CHUNK_SIZE: int = 1024 * 1024
DEFAULT_TIMEOUT: int = 60

DEFAULT_MINIMUM_YEAR: int = 1990

DEFAULT_EMBEDDER: str = "all-mpnet-base-v2"
DEFAULT_DEVICE: str = "cpu"

DEFAULT_THRESHOLD: float = 0.75
DEFAULT_GAP: float = 0.10

DEFAULT_BUFFER_SIZE: int = 1024 * 1024

# Paths

ROOT: Path = Path(__file__).resolve().parent.parent.parent

DATA_DIR: Path = ROOT / "data"

RAW_DIR: Path = DATA_DIR / "raw"

WIKTEXTRACT_PATH: Path = RAW_DIR / "wiktextract.jsonl.gz"

PROCESSED_DIR: Path = DATA_DIR / "processed"
SEED_PATH: Path = PROCESSED_DIR / "seed.jsonl"

# URLs

WIKTEXTRACT_URL: str = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"
