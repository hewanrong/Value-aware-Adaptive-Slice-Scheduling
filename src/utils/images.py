from __future__ import annotations

from pathlib import Path


def jpeg_size(path: str | Path) -> tuple[int, int]:
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    with Path(path).open("rb") as f:
        if f.read(2) != b"\xff\xd8":
            raise ValueError(f"Not a JPEG file: {path}")
        while True:
            byte = f.read(1)
            while byte and byte != b"\xff":
                byte = f.read(1)
            if not byte:
                break
            marker_byte = f.read(1)
            while marker_byte == b"\xff":
                marker_byte = f.read(1)
            if not marker_byte:
                break
            marker = marker_byte[0]
            if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
                continue
            length_bytes = f.read(2)
            if len(length_bytes) != 2:
                break
            marker_len = int.from_bytes(length_bytes, "big")
            if marker in sof_markers:
                payload = f.read(5)
                if len(payload) != 5:
                    break
                height = int.from_bytes(payload[1:3], "big")
                width = int.from_bytes(payload[3:5], "big")
                return width, height
            f.seek(max(0, marker_len - 2), 1)
    raise ValueError(f"Could not read JPEG dimensions from {path}")


def image_size(path: str | Path) -> tuple[int, int]:
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return jpeg_size(path)
    raise ValueError(f"Unsupported image type without optional imaging dependencies: {path}")
