"""Project paths and shared constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def model_short(name: str) -> str:
    """Short model name used in cache file names."""
    return name.rstrip("/").split("/")[-1]
