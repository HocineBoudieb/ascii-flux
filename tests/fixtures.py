"""Explicit analytical fixtures; no video, semantic training or D1 evidence."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from ascii_flux.contract import DEFAULT, TEXT_SPEC, canonical, digest
from ascii_flux.offline import ConverterConfig, convert_clip, converter_identity
from ascii_flux.storage import atomic_file, write_json, write_shard


def moving_blocks(contract):
    frames, masks = [], []
    for t in range(contract.frames):
        frame = np.zeros((48, 64, 3), np.uint8)
        mask = np.zeros((48, 64), np.uint8)
        x = 15 + t % 6
        mask[10:38, x:x+20] = 1
        frame[..., 0] = np.arange(64)[None]*4
        frame[..., 1] = np.arange(48)[:, None]*5
        frame[..., 2] = 70
        frames.append(frame)
        masks.append(mask)
    return frames, masks


def make_corpus(root, contract=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    contract = contract or replace(DEFAULT, version="analytical-fixture-1", cols=16, rows=8, frames=6)
    frames, masks = moving_blocks(contract)
    glyphs, colors, box, metrics = convert_clip(frames, masks, contract)
    source = {"source": "analytical-fixture", "conditions": "procedural test material; not the primary corpus"}
    converter = converter_identity(ConverterConfig())
    clips, records = [], []
    for i, (split, label) in enumerate((("train", "fixture a"), ("train", "fixture b"), ("val", "fixture a"), ("val", "fixture b"))):
        clip_id = f"fixture-{i}"
        meta = {"contract": contract.to_dict(), "normalized_label": label, "synthetic": True,
                "source": "analytical-fixture", "quality": metrics, "provenance": source, "converter": converter}
        clips.append((clip_id, glyphs, colors, box, meta))
        records.append({"clip_id": clip_id, "shard": "shard-00000.tar", "video_id": str(i), "source": "analytical-fixture",
                        "category": label, "normalized_label": label, "split": split, "synthetic": True, "quality": metrics})
    write_shard(root/"shard-00000.tar", clips, contract)
    from ascii_flux.factory import file_sha256
    manifest = {"dataset_version": "analytical-fixture-1", "contract": contract.to_dict(), "contract_hash": contract.hash,
                "source": source, "converter": converter, "config": {"heldout_categories": [], "quality": {"calibrated": False}},
                "clips": records, "shards": {"shard-00000.tar": file_sha256(root/"shard-00000.tar")},
                "report": {"D1_complete": False, "status": "analytical fixture; no semantic evidence"}}
    manifest["fingerprint"] = digest(manifest)
    write_json(root/"manifest.json", manifest)
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((2, 768)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    metadata = {"text_spec": TEXT_SPEC, "dataset_fingerprint": manifest["fingerprint"], "synthetic_encoder": True}
    with atomic_file(root/"embeddings.npz") as stream:
        np.savez_compressed(stream, labels=np.array(["fixture a", "fixture b"]), embeddings=vectors, metadata=np.array(canonical(metadata)))
    return root/"manifest.json", root/"embeddings.npz"

