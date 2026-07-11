from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator, Tuple


def iter_tfrecord_records(path: str | Path) -> Iterator[Tuple[int, bytes]]:
    p = Path(path)
    with p.open("rb") as f:
        idx = 0
        while True:
            header = f.read(12)
            if header == b"":
                break
            if len(header) != 12:
                raise IOError(f"truncated TFRecord header in {p} at record {idx}")
            length = struct.unpack("<Q", header[:8])[0]
            payload = f.read(length)
            if len(payload) != length:
                raise IOError(f"truncated TFRecord payload in {p} at record {idx}")
            crc = f.read(4)
            if len(crc) != 4:
                raise IOError(f"truncated TFRecord trailer in {p} at record {idx}")
            yield idx, payload
            idx += 1
