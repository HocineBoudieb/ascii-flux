"""Atomic immutable shards. No pickle is used for dataset material."""
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import numpy as np
from .contract import canonical


@contextmanager
def atomic_file(path, mode="wb"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, mode) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    with atomic_file(path) as stream:
        stream.write((canonical(value) + "\n").encode("utf-8"))


def validate_tokens(glyphs, colors, box, contract):
    expected = (contract.frames, contract.rows, contract.cols)
    if glyphs.shape != expected or colors.shape != expected or glyphs.dtype != np.uint8 or colors.dtype != np.uint8:
        raise ValueError("Tokens must be uint8 [T,R,C] matching the contract")
    if box.shape != (contract.frames, 4) or not np.isfinite(box).all():
        raise ValueError("Boxes must be finite [T,4]")
    if (box[:, 2:] <= box[:, :2]).any():
        raise ValueError("Invalid xyxy boxes")
    if glyphs.max() >= len(contract.glyphs):
        raise ValueError("Glyph outside vocabulary")
    occupied = glyphs != 0
    if np.any(colors[~occupied] != contract.empty_color) or np.any(colors[occupied] >= len(contract.palette)):
        raise ValueError("Invalid color, or a color on an empty cell")


def write_shard(path, clips, contract):
    """Publish tokens and their metadata together, after fsync."""
    with atomic_file(path) as stream:
        with tarfile.open(fileobj=stream, mode="w") as tar:
            for clip_id, glyphs, colors, boxes, metadata in clips:
                if not clip_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in clip_id):
                    raise ValueError("Unsafe clip ID")
                validate_tokens(glyphs, colors, boxes, contract)
                payload = io.BytesIO()
                np.savez_compressed(payload, glyphs=glyphs, colors=colors, box=boxes)
                for suffix, data in (("npz", payload.getvalue()), ("json", canonical(metadata).encode("utf-8"))):
                    info = tarfile.TarInfo(f"{clip_id}.{suffix}")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))


def read_clip(root, record, contract):
    root = Path(root).resolve()
    shard = (root / record["shard"]).resolve()
    if not shard.is_relative_to(root):
        raise ValueError("Shard path escapes dataset")
    with tarfile.open(shard, "r") as tar:
        clip_id = record["clip_id"]
        with tar.extractfile(f"{clip_id}.npz") as stream:
            with np.load(io.BytesIO(stream.read()), allow_pickle=False) as arrays:
                glyphs, colors, box = (arrays[key] for key in ("glyphs", "colors", "box"))
        metadata = json.loads(tar.extractfile(f"{clip_id}.json").read())
    contract.require(metadata["contract"])
    validate_tokens(glyphs, colors, box, contract)
    return glyphs, colors, box, metadata

