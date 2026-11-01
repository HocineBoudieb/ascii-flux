"""Official LV-VIS masked-video acquisition. Offline data source only."""
import copy
import json
from pathlib import Path
from .acquisition import download_file, extract_zip
from .factory import file_sha256
from .storage import write_json


def prepare_lvvis(data_root, profile):
    source = copy.deepcopy(profile["source"])
    if source["annotations_license"] != "CC-BY-NC-SA-4.0" or not source.get("terms_checked_on"):
        raise ValueError("Verify and record the original LV-VIS conditions before download")
    root = Path(data_root)/"lvvis"
    annotations = root/"downloads/train_instances.json"
    media = root/"downloads/train.zip"
    a = download_file(profile["annotations_url"], annotations, "annotations", expected_sha256=profile["annotations_sha256"])
    f = download_file(profile["frames_url"], media, "frames", expected_sha256=profile["frames_sha256"])
    frames = extract_zip(media, root/"frames", durable_files=False)
    normalized = root/"annotations.train.normalized.json"
    if not normalized.exists():
        data = json.loads(annotations.read_text(encoding="utf-8"))
        for video in data["videos"]:
            video["original_sequence_id"] = video["id"]
            video["id"] = "train-"+str(video["id"])
        for ann in data["annotations"]:
            ann["source_object_id"] = str(ann["id"])
            ann["id"] = "train-"+str(ann["id"])
            ann["video_id"] = "train-"+str(ann["video_id"])
        write_json(normalized, data)
    source["local_media_provenance"] = {"annotations": a, "frames": f}
    result = {"annotations": str(normalized.resolve()), "frames_root": str((frames/"train/JPEGImages").resolve()),
              "source_fps": profile["source_fps"], "source": source,
              "normalized_annotation_sha256": file_sha256(normalized)}
    write_json(root/"prepared.json", result)
    return result
