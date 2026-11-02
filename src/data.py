"""Load EuroSAT-MS (Sentinel-2, 13 bands) from the Hugging Face parquet shards into NumPy arrays."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

CLASSES = [
    "Annual Crop", "Forest", "Herbaceous Vegetation", "Highway", "Industrial Buildings",
    "Pasture", "Permanent Crop", "Residential Buildings", "River", "Sea/Lake",
]
AGRI_CLASSES = [0, 2, 5, 6]  # Annual Crop, Herbaceous Vegetation, Pasture, Permanent Crop

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
RGB = [3, 2, 1]  # B04, B03, B02

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"


def _read_shard(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["image", "label"])
    labels = np.asarray(table.column("label").to_numpy(), dtype=np.int64)
    flat = table.column("image").combine_chunks()
    # array3_d is stored as list<list<list<uint16>>>; unwrap the three list levels.
    values = flat.flatten().flatten().flatten().to_numpy(zero_copy_only=False)
    images = values.reshape(len(labels), 64, 64, 13).astype(np.uint16)
    return images, labels


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return images (N, 13, 64, 64) as uint16 and labels (N,) for 'train' or 'test'."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{split}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["x"], d["y"]
    shards = sorted(RAW.glob(f"{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No {split} shards in {RAW}. Run scripts/download.sh first.")
    xs, ys = zip(*(_read_shard(p) for p in shards))
    x = np.concatenate(xs).transpose(0, 3, 1, 2).copy()
    y = np.concatenate(ys)
    np.savez(cache, x=x, y=y)
    return x, y


def band_statistics(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean and std of reflectance, computed on training data only."""
    xf = x.astype(np.float32)
    return xf.mean(axis=(0, 2, 3)), xf.std(axis=(0, 2, 3)) + 1e-6


def sample_per_class(y: np.ndarray, n_per_class: int | None, seed: int) -> np.ndarray:
    """Indices of a class-balanced subset with n_per_class examples each (all if None)."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        members = np.flatnonzero(y == c)
        rng.shuffle(members)
        idx.append(members if n_per_class is None else members[:n_per_class])
    out = np.concatenate(idx)
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    for s in ("train", "test"):
        x, y = load_split(s)
        counts = np.bincount(y, minlength=len(CLASSES))
        print(s, x.shape, x.dtype, dict(zip(CLASSES, counts.tolist())))
    os.environ.setdefault("PYTHONHASHSEED", "0")
