"""YouTube-VIS adapter and auditable D1 run. Reads local media only."""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import numpy as np
from PIL import Image
from .contract import DEFAULT, digest, normalize_text
from .offline import ConverterConfig, convert_clip, converter_identity, quality_rejections
from .storage import write_json, write_shard


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def validate_source(source):
    for key in ("source", "snapshot", "url", "terms_url", "terms_checked_on", "annotations_license", "media_conditions", "local_media_provenance"):
        value = source.get(key, "")
        if not value or "REPLACE_" in value:
            raise ValueError(f"Complete source provenance before conversion: {key}")
    if source.get("source") != "YouTube-VIS" or source.get("usage_scope") != "non-commercial-research":
        raise ValueError("This adapter is restricted to the recorded YouTube-VIS research conditions")


def decode_mask(segmentation, height, width):
    if not segmentation:
        return np.zeros((height, width), np.uint8)
    if isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
        if segmentation.get("size", [height, width]) != [height, width]:
            raise ValueError("invalid_mask_dimensions")
        counts = segmentation["counts"]
        if any(not isinstance(n, int) or n < 0 for n in counts) or sum(counts) != height*width:
            raise ValueError("invalid_mask_rle")
        flat = np.zeros(height*width, np.uint8)
        pos, bit = 0, 0
        for length in counts:
            flat[pos:pos+length] = bit
            pos += length
            bit = 1-bit
        return flat.reshape((height, width), order="F")
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError("Compressed YouTube-VIS RLE requires pip install '.[vis]'") from exc
    if isinstance(segmentation, list):
        segmentation = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    result = mask_utils.decode(segmentation)
    if result.shape != (height, width):
        raise ValueError("invalid_mask_dimensions")
    return result.astype(np.uint8)


def split_videos(videos, annotations, categories, heldout, seed):
    missing = set(heldout) - set(categories.values())
    if missing:
        raise ValueError(f"Unknown held-out categories in this release: {sorted(missing)}")
    reserved = {a["video_id"] for a in annotations if categories[a["category_id"]] in heldout}
    result = {}
    for v in videos:
        if v in reserved:
            result[v] = "test_unseen"
        else:
            number = int(hashlib.sha256(f"{seed}:YouTube-VIS:{v}".encode()).hexdigest()[:12], 16) / 16**12
            result[v] = "train" if number < .8 else "val" if number < .9 else "test_seen"
    return result


def build_dataset(annotations_path, frames_root, source, config, output, source_fps, contract=DEFAULT):
    validate_source(source)
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Supply the actual cadence of the extracted annotated frames, not the original video's FPS")
    if config["max_clips"] <= 0 or config["shard_size"] <= 0:
        raise ValueError("Invalid run limits")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Dataset runs are immutable: {output}")
    root = Path(frames_root).resolve()
    data = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    videos = {v["id"]: v for v in data["videos"]}
    categories = {c["id"]: c["name"] for c in data["categories"]}
    partitions = split_videos(videos, data["annotations"], categories, config["heldout_categories"], config["seed"])
    converter = ConverterConfig(**config["converter"])
    identity = converter_identity(converter)
    annotation_hash = file_sha256(annotations_path)
    # Round-robin categories after seeded shuffling, rather than taking the first 200 annotations.
    grouped = defaultdict(list)
    for ann in data["annotations"]:
        grouped[ann["category_id"]].append(ann)
    rng = random.Random(config["seed"])
    for items in grouped.values():
        rng.shuffle(items)
    ordered = []
    while any(grouped.values()):
        for category in sorted(grouped):
            if grouped[category]:
                ordered.append(grouped[category].pop())
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent))
    records, rejects, pending = [], [], []
    shard_number = 0

    def flush():
        nonlocal shard_number
        if pending:
            write_shard(staging / f"shard-{shard_number:05d}.tar", pending, contract)
            pending.clear()
            shard_number += 1

    try:
        attempted = 0
        for ann in ordered:
            v = videos[ann["video_id"]]
            n = len(v["file_names"])
            # Only full real-duration clips. Never silently loop or pad a short video.
            duration = n/source_fps
            starts = np.arange(0, max(0., duration-contract.frames/contract.fps)+1e-8, contract.frames/contract.fps)
            if duration + 1e-8 < contract.frames/contract.fps:
                rejects.append({"video_id": str(v["id"]), "instance_id": str(ann["id"]), "source": source["source"], "reasons": ["too_short_for_48_frames"]})
                continue
            for start in starts:
                if attempted >= config["max_clips"]:
                    break
                attempted += 1
                clip_id = f"v{v['id']}-i{ann['id']}-t{round(start*1000)}"
                base = {"clip_id": clip_id, "video_id": str(v["id"]), "instance_id": str(ann["id"]),
                        "source": source["source"], "category": categories[ann["category_id"]], "split": partitions[v["id"]]}
                indices = np.floor((start+np.arange(contract.frames)/contract.fps)*source_fps + 1e-7).astype(int)
                try:
                    frames, masks, hashes = [], [], {}
                    for index in indices:
                        path = (root / v["file_names"][index]).resolve()
                        if not path.is_relative_to(root):
                            raise ValueError("media_path_escapes_root")
                        with Image.open(path) as image:
                            frame = np.array(image.convert("RGB"))
                        if frame.shape[:2] != (v["height"], v["width"]):
                            raise ValueError("source_dimensions_mismatch")
                        if index >= len(ann["segmentations"]):
                            raise ValueError("missing_annotation_frame")
                        frames.append(frame)
                        masks.append(decode_mask(ann["segmentations"][index], v["height"], v["width"]))
                        hashes[str(index)] = file_sha256(path)
                    glyphs, colors, boxes, metrics = convert_clip(frames, masks, contract, converter)
                    reasons = quality_rejections(metrics, config["quality"])
                    if reasons:
                        rejects.append({**base, "reasons": reasons, "quality": metrics})
                        continue
                    label = normalize_text(base["category"])
                    metadata = {**base, "provenance": source, "annotation_sha256": annotation_hash,
                                "source_frame_sha256": hashes, "source_frame_indices": indices.tolist(),
                                "source_frame_fps": source_fps, "sampled_unique_source_frames": len(set(indices.tolist())),
                                "source_frame_paths": [v["file_names"][index] for index in indices],
                                "source_frame_indices_30fps": [v["source_frame_indices_30fps"][index] for index in indices] if "source_frame_indices_30fps" in v else None,
                                "original_sequence_id": v.get("original_sequence_id"),
                                "source_object_id": ann.get("source_object_id"),
                                "source_url": v.get("url", source["url"]), "start_seconds": float(start),
                                "contract": contract.to_dict(), "contract_hash": contract.hash,
                                "converter": identity, "quality": metrics, "mask_method": "source-instance-mask",
                                "original_label": base["category"], "original_caption": v.get("caption"),
                                "normalized_label": label, "label_scope": "subject only; action not annotated by YouTube-VIS",
                                "rewrite": None, "synthetic": bool(source.get("synthetic", False)), "human_review": "pending",
                                "box_format": "xyxy source pixels; outside-image padding permitted"}
                    pending.append((clip_id, glyphs, colors, boxes, metadata))
                    records.append({**base, "shard": f"shard-{shard_number:05d}.tar", "normalized_label": label,
                                    "synthetic": bool(source.get("synthetic", False)), "quality": metrics})
                    if len(pending) >= config["shard_size"]:
                        flush()
                except (ValueError, OSError, IndexError) as exc:
                    rejects.append({**base, "reasons": [type(exc).__name__ + ":" + str(exc)]})
            if attempted >= config["max_clips"]:
                break
        flush()
        counts = Counter(r["split"] for r in records)
        report = {"source": source, "attempted_full_clips": attempted, "machine_accepted": len(records),
                  "rejected_records": len(rejects), "yield_full_clips": len(records)/max(1, attempted),
                  "yield_including_short_rejections": len(records)/max(1, len(records)+len(rejects)),
                  "split_counts": dict(counts), "reasons": dict(Counter(reason for r in rejects for reason in r["reasons"])),
                  "quality_thresholds_calibrated": config["quality"].get("calibrated", False),
                  "human_review_completed": False, "D1_complete": False,
                  "status": "candidates for human review; not a validated corpus"}
        manifest = {"dataset_version": config["dataset_version"], "contract": contract.to_dict(),
                    "contract_hash": contract.hash, "source": source, "converter": identity,
                    "annotation_sha256": annotation_hash, "config": config, "clips": records,
                    "shards": {p.name: file_sha256(p) for p in staging.glob("*.tar")},
                    "video_splits": {str(k): v for k, v in partitions.items()}, "report": report}
        manifest["fingerprint"] = digest(manifest)
        write_json(staging/"manifest.json", manifest)
        write_json(staging/"rejections.json", rejects)
        write_json(staging/"report.json", report)
        write_json(staging/"reviews.template.json", {"dataset_fingerprint": manifest["fingerprint"], "reviewer": "",
                    "converter_hash": identity["code_sha256"], "decisions": {r["clip_id"]: {"decision": "pending", "reason": ""} for r in records}})
        if output.exists():
            raise FileExistsError(f"Output appeared while building this immutable run: {output}")
        os.replace(staging, output)
        return manifest
    except BaseException:
        resolved = staging.resolve()
        if resolved.parent == output.parent and resolved.name.startswith(output.name + "."):
            shutil.rmtree(resolved)
        raise
