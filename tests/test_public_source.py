import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile
import pytest
from ascii_flux.acquisition import download_file, load_prepared
from ascii_flux.public_source import normalize_repacked_annotations, extract_selected_tar, prepare_public_ytvis2019
from tests.test_acquisition import source_template, zip_bytes


def annotation_repack():
    return {"meta": {"category_labels": {"1": "fixture-only"}}, "sequences": [
        {"id": 949, "height": 2, "width": 2, "length": 24,
         "image_paths": [f"JPEGImages/video-a/{i*5:05d}.jpg" for i in range(24)],
         "categories": {"7": 1}, "segmentations": [{"7": "04"}]*24}]}


def tar_bytes(entries):
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w") as archive:
        for name, content in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return result.getvalue()


def test_repack_preserves_labels_rle_source_ids_and_records_exclusions():
    raw = annotation_repack()
    short = copy.deepcopy(raw["sequences"][0])
    short.update(id=2, length=3, image_paths=[f"JPEGImages/short/{i*5:05d}.jpg" for i in range(3)], segmentations=[{"7":"04"}]*3)
    irregular = copy.deepcopy(raw["sequences"][0])
    irregular["image_paths"] = [p.replace("video-a", "irregular") for p in irregular["image_paths"]]
    irregular["image_paths"][-1] = "JPEGImages/irregular/00120.jpg"
    raw["sequences"] += [short, irregular]
    result, report = normalize_repacked_annotations(raw)
    assert [v["id"] for v in result["videos"]] == ["video-a"]
    assert result["videos"][0]["original_sequence_id"] == 949
    assert result["videos"][0]["source_frame_indices_30fps"] == list(range(0,120,5))
    ann = result["annotations"][0]
    assert ann["source_object_id"] == "7" and ann["id"] == "video-a-o7"
    assert ann["segmentations"][0] == {"size": [2, 2], "counts": "04"}
    assert result["categories"] == [{"id": 1, "name": "fixture-only"}]
    assert report["input_videos"] == 3 and report["eligible_videos"] == 1
    assert report["excluded"][0]["reasons"] == ["shorter_than_target_clip"]
    assert report["excluded"][1]["reasons"] == ["irregular_annotated_frame_cadence"]


def test_public_profile_complete_preparation_without_account_or_input(tmp_path):
    from PIL import Image
    raw = annotation_repack()
    ann_bytes = zip_bytes({"youtube_vis_train.json": json.dumps(raw).encode()})
    image = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(image, format="JPEG")
    entries = {"train/"+p: image.getvalue() for p in raw["sequences"][0]["image_paths"]}
    entries["train/unrelated.txt"] = b"not part of VIS"
    media_bytes = tar_bytes(entries)
    profile = {"profile": "analytical-fixture-public-profile", "checked_on": "2026-10-31",
               "annotations": {"url": "https://example.invalid/annotations", "sha256": hashlib.sha256(ann_bytes).hexdigest()},
               "frames": {"url": "https://example.invalid/frames", "sha256": hashlib.sha256(media_bytes).hexdigest(), "bytes": len(media_bytes)},
               "terms_url": "https://youtube-vos.org/dataset/term/", "cadence_evidence": "fixture", "source_fps": 6.}
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs["url"])
        Path(kwargs["output"]).write_bytes(ann_bytes if kwargs["url"].endswith("annotations") else media_bytes)
        return kwargs["output"]

    result = prepare_public_ytvis2019(tmp_path, source_template(), profile, downloader)
    assert result["videos"] == 1 and result["instances"] == 1 and result["listed_frames"] == 24
    assert result["source"]["synthetic"]
    assert result["source"]["usage_scope"] == "non-commercial-research"
    assert result == load_prepared(tmp_path, profile="ytvis-2019-public")
    assert not (Path(result["frames_root"])/"unrelated.txt").exists()
    assert prepare_public_ytvis2019(tmp_path, source_template(), profile, downloader) == result
    assert calls == [profile["annotations"]["url"], profile["frames"]["url"]]


def test_changed_mirror_bytes_fail_before_publication(tmp_path):
    payload = zip_bytes({"file": b"fixture"})
    def downloader(**kwargs):
        Path(kwargs["output"]).write_bytes(payload)
        return kwargs["output"]
    with pytest.raises(ValueError, match="SHA-256"):
        download_file("https://example.invalid/changed", tmp_path/"archive.zip", "frames", downloader, expected_sha256="0"*64)
    assert not (tmp_path/"archive.zip").exists()


@pytest.mark.parametrize("name,kind", [("../outside", "file"), ("safe/link", "link")])
def test_tar_rejects_traversal_and_links_before_publication(tmp_path, name, kind):
    archive_path = tmp_path/"unsafe.tar"
    with tarfile.open(archive_path, "w") as archive:
        info = tarfile.TarInfo(name)
        if kind == "link":
            info.type = tarfile.SYMTYPE
            info.linkname = "../../outside"
        archive.addfile(info)
    with pytest.raises(ValueError):
        extract_selected_tar(archive_path, tmp_path/"images", {name})
    assert not (tmp_path/"images").exists()


def test_notebook_uses_public_profile_without_login_placeholders():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root/"ASCII_Flux_v5_D1_F1.ipynb").read_text(encoding="utf-8"))
    acquire = "".join(next(c for c in notebook["cells"] if c["id"] == "acquire-v5")["source"])
    assert "prepare_public_ytvis2019" in acquire
    assert "input(" not in acquire and "PORTAL" not in acquire
    profile = json.loads((root/"configs/ytvis2019_public.json").read_text())
    assert profile["annotations"]["version"] == 1
    assert len(profile["frames"]["revision"]) == 40 and len(profile["frames"]["sha256"]) == 64
