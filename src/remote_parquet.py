"""Read selected row groups of a remote Parquet file over HTTP range requests (no full download)."""

from __future__ import annotations

import io
import time
import urllib.request

import pyarrow.parquet as pq

BASE = "https://huggingface.co/datasets/blanchon/EuroSAT_MSI/resolve/main/data"


class HTTPRangeFile(io.RawIOBase):
    """Minimal seekable, read-only file over HTTP Range requests, with retries."""

    def __init__(self, url: str, retries: int = 20):
        self.url, self.pos, self.retries = url, 0, retries
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.fetched = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        end = self.pos + n - 1
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) == n:
                    break
            except Exception:
                time.sleep(min(30, 2 ** attempt))
        else:
            raise IOError(f"failed to read {self.url} [{self.pos}-{end}]")
        self.pos += len(data)
        self.fetched += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_remote(name: str) -> tuple[pq.ParquetFile, HTTPRangeFile]:
    f = HTTPRangeFile(f"{BASE}/{name}.parquet")
    return pq.ParquetFile(f), f
