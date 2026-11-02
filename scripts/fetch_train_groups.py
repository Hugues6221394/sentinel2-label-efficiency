"""
Fetch only the training row groups needed for >= 250 images of every class.

The EuroSAT-MS training split on Hugging Face is stored in four Parquet shards sorted by class.
Shard 0 (downloaded in full by download.sh) covers Annual Crop, Forest and Herbaceous Vegetation;
the row groups below cover the remaining seven classes. Each is saved as a small local Parquet file.
"""

import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from remote_parquet import open_remote  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
GROUPS = [("train-00001-of-00004", 2), ("train-00002-of-00004", 0), ("train-00002-of-00004", 1),
          ("train-00003-of-00004", 0), ("train-00003-of-00004", 2)]

for shard, g in GROUPS:
    out = RAW / f"{shard}-g{g}.parquet"
    if out.exists():
        print("exists", out.name, flush=True)
        continue
    pf, f = open_remote(shard)
    table = pf.read_row_group(g, columns=["image", "label"])
    tmp = out.with_suffix(".tmp")
    pq.write_table(table, tmp)
    tmp.rename(out)
    print(f"saved {out.name}: {table.num_rows} rows, fetched {f.fetched / 1e6:.0f} MB", flush=True)
