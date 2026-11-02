#!/usr/bin/env bash
# Downloads the EuroSAT-MS data used in this study (about 1.3 GB): one full training shard, five training
# row groups covering the remaining classes, and the full test split
# from the Hugging Face mirror of EuroSAT (Helber et al., 2019).
set -euo pipefail
cd "$(dirname "$0")/../data/raw"
base="https://huggingface.co/datasets/blanchon/EuroSAT_MSI/resolve/main/data"
for f in train-00000-of-00004 test-00000-of-00002 test-00001-of-00002; do
  curl -L -C - --retry 20 --retry-delay 3 --retry-all-errors -o "$f.parquet" "$base/$f.parquet"
done
# Training shards 1-3 are sorted by class; fetch only the row groups needed for the remaining classes.
cd ../..
python scripts/fetch_train_groups.py
