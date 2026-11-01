"""Bind sampled human converter review to an immutable training snapshot."""
import copy
from pathlib import Path
from .contract import digest
from .offline import quality_rejections
from .review import load_review_manifest
from .storage import write_json


def calibrated_view(manifest_path, output, quality):
    manifest = copy.deepcopy(load_review_manifest(manifest_path))
    if Path(output).resolve().parent != Path(manifest_path).resolve().parent:
        raise ValueError("Keep a calibrated view beside its immutable shards")
    if not quality.get("calibrated"):
        raise ValueError("An explicit calibrated policy is required")
    original = manifest["fingerprint"]
    removed = [r for r in manifest["clips"] if quality_rejections(r["quality"], quality)]
    manifest["clips"] = [r for r in manifest["clips"] if not quality_rejections(r["quality"], quality)]
    manifest["config"]["quality"] = quality
    manifest["dataset_version"] += "-calibrated-view-1"
    manifest["parent_dataset_fingerprint"] = original
    manifest["report"].update(machine_accepted=len(manifest["clips"]), calibration_rejected=len(removed),
                              human_review_scope="global converter sample review; not individual acceptance of every clip")
    manifest.pop("fingerprint")
    manifest["fingerprint"] = digest(manifest)
    target = Path(output)
    if target.exists() and load_review_manifest(target) != manifest:
        raise FileExistsError("Calibrated views are immutable")
    write_json(target, manifest)
    return target


def bind_sample_review(manifest_path, sample_manifest_path, output, reviewer, human_statement):
    manifest, sample = load_review_manifest(manifest_path), load_review_manifest(sample_manifest_path)
    if manifest["converter"] != sample["converter"] or manifest["contract"] != sample["contract"]:
        raise ValueError("Review sample must use the exact converter and representation")
    quality = manifest["config"]["quality"]
    if not quality.get("calibrated") or not reviewer.strip() or not human_statement.strip():
        raise ValueError("Need calibrated policy and an actual identified human review")
    if quality.get("calibration_sample_fingerprint") != sample["fingerprint"]:
        raise ValueError("Quality policy was calibrated on a different human review sample")
    value = {"scope": "converter-sample", "dataset_fingerprint": manifest["fingerprint"],
             "converter_hash": manifest["converter"]["code_sha256"], "converter_identity": manifest["converter"],
             "quality_policy": quality, "reviewer": reviewer, "human_statement": human_statement,
             "sample_dataset_fingerprint": sample["fingerprint"], "sample_available_clips": len(sample["clips"]),
             "number_of_clips_actually_inspected": None, "per_clip_human_acceptance": False,
             "decisions": {}, "note": "Automatically filtered clips; sample review does not certify every clip"}
    write_json(output, value)
    return Path(output)
