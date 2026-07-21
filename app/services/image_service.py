import hashlib
from pathlib import Path
from typing import Iterable

from PIL import Image
import imagehash

from app.core.config import SUPPORTED_EXTENSIONS


def iter_images(folder: Path) -> Iterable[Path]:
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: Path) -> str:
    with Image.open(path) as image:
        return str(imagehash.phash(image.convert("RGB")))


def image_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def simple_category(path: Path) -> tuple[str, str]:
    """Lightweight MVP categorizer; replace with CLIP in the advanced version."""
    name = path.name.lower()
    rules = {
        "receipt": ["receipt", "invoice", "bill"],
        "prescription": ["prescription", "medicine", "medical"],
        "document": ["document", "passport", "certificate", "form"],
        "travel": ["travel", "trip", "beach", "mountain", "hotel"],
        "pet": ["dog", "cat", "pet"],
        "people": ["selfie", "portrait", "person", "people", "family"],
    }
    for category, keywords in rules.items():
        if any(keyword in name for keyword in keywords):
            return category, f"Image categorized as {category} using filename-based MVP rules."
    return "other", "Image categorized as other using filename-based MVP rules."
