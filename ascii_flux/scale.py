"""Resumable D2 conversion in bounded, independently committed shard jobs."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image
from .catalogue import Catalogue, writer_lock
from .contract import DEFAULT, digest, normalize_text
from .factory import decode_mask, file_sha256, split_videos, validate_source
from .offline import ConverterConfig, convert_clip, converter_identity, quality_rejections
from .storage import write_json, write_shard


def inventory(annotations_path, source_fps, contract=DEFAULT):
    if not source_fps or not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Record and verify the annotated frame cadence before conversion")
    data = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    videos = {v["id"]: v for v in data["videos"]}
    clips = sum(int((len(videos[a['video_id']]['file_names'])/source_fps+1e-8)//(contract.frames/contract.fps))
                for a in data["annotations"])
    return {"videos": len(videos), "instances": len(data["annotations"]), "nonoverlapping_full_clips_upper_bound": clips,
            "grid_slots_upper_bound": clips*contract.frames, "source_frame_fps": source_fps,
            "note": "upper bound before missing masks, quality rejection and duplicate checks"}


def _convert_job(tasks, frames_root, source, config, source_fps, contract, annotation_hash, output):
    output = Path(output)
    if (output/"manifest.json").exists():
        return output/"manifest.json"
    output.mkdir(parents=True, exist_ok=True)
    converter = ConverterConfig(**config["converter"])
    identity = converter_identity(converter)
    root = Path(frames_root).resolve()
    clips, records, rejects = [], [], []
    for ann, video, category, split, start in tasks:
        clip_id = "clip-"+digest([source["source"], str(video["id"]), str(ann["id"]), start])[:32]
        base = {"clip_id": clip_id, "video_id": str(video["id"]), "instance_id": str(ann["id"]),
                "source": source["source"], "category": category, "split": split}
        indices = np.floor((start+np.arange(contract.frames)/contract.fps)*source_fps+1e-7).astype(int)
        try:
            # Decode repeated source frames once; conversion still processes all
            # time steps and retains the same stabilizer as the D1 converter.
            decoded, hashes = {}, {}
            for index in sorted(set(indices.tolist())):
                path = (root/video["file_names"][index]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("media_path_escapes_root")
                with Image.open(path) as image:
                    frame = np.array(image.convert("RGB"))
                if frame.shape[:2] != (video["height"], video["width"]):
                    raise ValueError("source_dimensions_mismatch")
                mask = decode_mask(ann["segmentations"][index], video["height"], video["width"])
                decoded[index] = (frame, mask)
                hashes[str(index)] = file_sha256(path)
            glyphs, colors, boxes, metrics = convert_clip([decoded[i][0] for i in indices],
                                                         [decoded[i][1] for i in indices], contract, converter)
            reasons = quality_rejections(metrics, config["quality"])
            if reasons:
                rejects.append({**base, "reasons": reasons, "quality": metrics})
                continue
            label = normalize_text(category)
            metadata = {**base, "provenance": source, "annotation_sha256": annotation_hash,
                        "source_video_uid": f"{source['source']}:{video['id']}",
                        "source_frame_sha256": hashes, "source_frame_indices": indices.tolist(),
                        "source_frame_paths": [video["file_names"][i] for i in indices],
                        "source_frame_fps": source_fps, "sampled_unique_source_frames": len(decoded),
                        "source_frame_indices_30fps": [video["source_frame_indices_30fps"][i] for i in indices]
                            if "source_frame_indices_30fps" in video else None,
                        "source_object_id": ann.get("source_object_id"), "original_sequence_id": video.get("original_sequence_id"),
                        "source_url": video.get("url", source["url"]), "start_seconds": start,
                        "contract": contract.to_dict(), "contract_hash": contract.hash, "converter": identity,
                        "quality": metrics, "mask_method": "source-instance-mask", "original_label": category,
                        "original_caption": video.get("caption"), "normalized_label": label,
                        "label_scope": "subject only; action not annotated", "rewrite": None,
                        "synthetic": bool(source.get("synthetic", False)), "human_review": "pending",
                        "box_format": "xyxy source pixels; outside-image padding permitted"}
            clips.append((clip_id, glyphs, colors, boxes, metadata))
            records.append({**base, "shard": "shard-00000.tar", "normalized_label": label,
                            "synthetic": metadata["synthetic"], "quality": metrics})
            if config.get("horizontal_flip", False):
                if contract.glyphs != " .:+*#@":
                    raise ValueError("Horizontal augmentation needs a verified mirror mapping for contour glyphs")
                augmented_id = clip_id+"-flip"
                augmentation = {"kind": "horizontal_flip", "parent_clip_id": clip_id,
                                "new_source_observations": False,
                                "box_coordinates": "original source trajectory; rendered crop is mirrored"}
                augmented_meta = dict(metadata, clip_id=augmented_id, augmentation=augmentation)
                clips.append((augmented_id, glyphs[..., ::-1].copy(), colors[..., ::-1].copy(), boxes, augmented_meta))
                records.append(dict(records[-1], clip_id=augmented_id, augmentation=augmentation))
        except (ValueError, OSError, IndexError) as error:
            rejects.append({**base, "reasons": [type(error).__name__+":"+str(error)]})
    shards = {}
    if clips:
        write_shard(output/"shard-00000.tar", clips, contract)
        shards["shard-00000.tar"] = file_sha256(output/"shard-00000.tar")
    report = {"attempted_full_clips": len(tasks), "machine_accepted": len(records), "rejected_records": len(rejects),
              "reasons": dict(Counter(reason for r in rejects for reason in r["reasons"])),
              "split_counts": dict(Counter(r["split"] for r in records)), "human_review_completed": False,
              "D1_complete": False, "status": "D2 machine candidates; no human recognition claim"}
    manifest = {"dataset_version": config["dataset_version"], "contract": contract.to_dict(),
                "contract_hash": contract.hash, "source": source, "converter": identity,
                "annotation_sha256": annotation_hash, "config": config, "clips": records, "shards": shards,
                "report": report, "adapter_sha256": file_sha256(__file__)}
    manifest["fingerprint"] = digest(manifest)
    write_json(output/"rejections.json", rejects)
    write_json(output/"manifest.json", manifest)  # commit marker, written last
    return output/"manifest.json"


def build_scale(annotations_path, frames_root, source, config, output, source_fps, workers=2, max_jobs=None,
                contract=DEFAULT):
    if source["source"] == "YouTube-VIS":
        validate_source(source)
    elif source["source"] == "LV-VIS":
        for key in ("snapshot", "url", "terms_url", "terms_checked_on", "annotations_license", "media_conditions", "cadence_evidence"):
            if not source.get(key):
                raise ValueError("Missing LV-VIS provenance: "+key)
        if source["annotations_license"] != "CC-BY-NC-SA-4.0" or source.get("usage_scope") != "non-commercial-research":
            raise ValueError("Retain LV-VIS original noncommercial research conditions")
    else:
        raise ValueError("No verified masked-video adapter for this source")
    if workers < 1 or config["shard_size"] < 1 or (max_jobs is not None and max_jobs < 1):
        raise ValueError("Positive worker and job limits required")
    summary = inventory(annotations_path, source_fps, contract)
    output = Path(output).resolve()
    identity = {"annotation_sha256": file_sha256(annotations_path), "source": source, "config": config,
                "contract": contract.to_dict(), "converter": converter_identity(ConverterConfig(**config["converter"])),
                "adapter_sha256": file_sha256(__file__), "source_fps": source_fps,
                "frames_root": str(Path(frames_root).resolve())}
    plan_id = digest(identity)
    with writer_lock(output), Catalogue(output, contract, config.get("target_grids", 1_000_000)) as catalogue:
        plan = output/"plans"/plan_id
        plan.mkdir(parents=True, exist_ok=True)
        write_json(plan/"plan.json", {**identity, "inventory": summary})
        data = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
        videos = {v["id"]: v for v in data["videos"]}
        categories = {c["id"]: c["name"] for c in data["categories"]}
        # YouTube-VIS preserves exactly the D1 partition rule across chunks.
        splits = split_videos(videos, data["annotations"], categories, config["heldout_categories"], config["seed"])
        tasks = []
        seconds = contract.frames/contract.fps
        for ann in sorted(data["annotations"], key=lambda a: str(a["id"])):
            video = videos[ann["video_id"]]
            count = int((len(video["file_names"])/source_fps+1e-8)//seconds)
            for i in range(count):
                tasks.append((ann, video, categories[ann["category_id"]], splits[video["id"]], i*seconds))
        jobs = [tasks[i:i+config["shard_size"]] for i in range(0, len(tasks), config["shard_size"])]
        pending = []
        for number, batch in enumerate(jobs):
            marker = plan/f"job-{number:06d}"/"manifest.json"
            if marker.exists():
                # Re-ingest after a crash between file publication and SQL commit.
                catalogue.ingest(marker)
            else:
                pending.append((number, batch))
        # A bounded map keeps at most `workers` full frame buffers in memory.
        import cv2
        cv2.setNumThreads(1)
        start = time.monotonic()
        def work(item):
            number, batch = item
            return _convert_job(batch, frames_root, source, config, source_fps, contract,
                                identity["annotation_sha256"], plan/f"job-{number:06d}")
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Explicit bounded submission (Executor.map queues everything on
            # Python 3.11/3.12). Finish each window before submitting the next.
            for first in range(0, len(pending), workers):
                window = pending[first:first+workers]
                if max_jobs is not None:
                    window = window[:max(0, max_jobs-completed)]
                if not window:
                    break
                for manifest_path in pool.map(work, window):
                    catalogue.ingest(manifest_path)
                    completed += 1
                    report = catalogue.report()
                    report.update(jobs_completed_this_call=completed, jobs_total=len(jobs),
                                  elapsed_seconds=round(time.monotonic()-start, 1), inventory=summary)
                    write_json(output/"progress.json", report)
                    print(json.dumps({k: report[k] for k in ("clips", "grid_slots", "distinct_token_grids", "jobs_completed_this_call", "jobs_total")}), flush=True)
        report = catalogue.report()
        write_json(output/"report.json", report)
        return report

