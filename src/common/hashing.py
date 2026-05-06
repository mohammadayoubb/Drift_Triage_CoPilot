# src/common/hashing.py

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """
    Compute a SHA256 hash for a file.

    We use this in the model card so we can prove which raw dataset
    and which model artifact produced a training run.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {path}")

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()