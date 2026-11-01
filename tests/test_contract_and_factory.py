from dataclasses import replace
import json
import numpy as np
import pytest
from PIL import Image
from ascii_flux.contract import DEFAULT, Contract
from ascii_flux.factory import build_dataset, decode_mask, split_videos, validate_source
from ascii_flux.offline import convert_clip, quality_rejections, tracked_boxes, ConverterConfig
from ascii_flux.storage import atomic_file, read_clip
from tests.fixtures import make_corpus, moving_blocks


def test_contract_changes_invalidate_weights():
    for changed in (replace(DEFAULT, glyphs=" .:+*@#"), replace(DEFAULT, fps=24),
                    replace(DEFAULT, cell_height_width=1.8), replace(DEFAULT, palette=DEFAULT.palette[::-1])):
        with pytest.raises(ValueError, match="Incompatible"):
            DEFAULT.require(changed.to_dict())
    assert Contract.from_dict(DEFAULT.to_dict()).hash == DEFAULT.hash


def test_masks_rle_and_compressed_parity():
    mask = decode_mask({"counts": [2, 3, 7], "size": [3, 4]}, 3, 4)
    assert mask.sum() == 3 and mask[2, 0] == 1 and mask[0, 1] == 1
    with pytest.raises(ValueError):
        decode_mask({"counts": [2, 99]}, 3, 4)
    mask_utils = pytest.importorskip("pycocotools.mask")
    encoded = mask_utils.encode(np.asfortranarray(mask))
    np.testing.assert_array_equal(decode_mask(encoded, 3, 4), mask)


def test_split_all_instances_of_reserved_video_are_held_out():
    anns = [{"video_id": 1, "category_id": 1}, {"video_id": 1, "category_id": 2}, {"video_id": 2, "category_id": 2}]
    categories = {1: "reserved", 2: "seen"}
    splits = split_videos({1: {}, 2: {}}, anns, categories, ["reserved"], 42)
    assert splits[1] == "test_unseen"
    assert splits == split_videos({2: {}, 1: {}}, anns[::-1], categories, ["reserved"], 42)
    with pytest.raises(ValueError, match="Unknown"):
        split_videos({1: {}}, anns[:1], categories, ["typo"], 42)


def test_converter_empty_colors_crop_aspect_and_temporal_stability():
    contract = replace(DEFAULT, cols=16, rows=8, frames=6)
    frames, masks = moving_blocks(contract)
    glyphs, colors, boxes, metrics = convert_clip(frames, masks, contract)
    assert glyphs.shape == (6, 8, 16) and glyphs.dtype == np.uint8
    assert np.all(colors[glyphs == 0] == 255)
    assert np.all(colors[glyphs != 0] < 64)
    aspect = (boxes[:, 2]-boxes[:, 0])/(boxes[:, 3]-boxes[:, 1])
    np.testing.assert_allclose(aspect, contract.cols/(contract.rows*contract.cell_height_width), rtol=1e-6)
    assert metrics["mask_fraction"] == 1
    # Stationary identical inputs should not flicker after stabilization.
    g, _, _, m = convert_clip([frames[0]]*6, [masks[0]]*6, contract)
    assert not np.any(g[1:] != g[:-1]) and m["glyph_change_compensated_active"] == 0
    assert quality_rejections(metrics, {"min_occupancy": 1}) == ["min_occupancy"]
    with pytest.raises(ValueError, match="no_subject_mask"):
        tracked_boxes([np.zeros_like(masks[0])]*6, contract, ConverterConfig())


def test_atomic_replacement_survives_interrupted_write(tmp_path):
    path = tmp_path/"state.json"
    path.write_bytes(b"old")
    with pytest.raises(RuntimeError):
        with atomic_file(path) as f:
            f.write(b"partial")
            raise RuntimeError("interruption")
    assert path.read_bytes() == b"old"
    assert not list(tmp_path.glob("*.tmp"))


def test_shard_roundtrip_and_no_path_escape(tmp_path):
    manifest_path, _ = make_corpus(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    contract = Contract.from_dict(manifest["contract"])
    g, c, b, meta = read_clip(tmp_path, manifest["clips"][0], contract)
    assert meta["synthetic"] and b.shape == (contract.frames, 4)
    with pytest.raises(ValueError, match="escapes"):
        read_clip(tmp_path, {"shard": "../outside.tar"}, contract)


def test_official_format_adapter_and_report(tmp_path):
    frames_root = tmp_path/"media"
    frames_root.mkdir()
    contract = replace(DEFAULT, frames=6, cols=16, rows=8, fps=3)
    frames, masks = moving_blocks(contract)
    names = []
    segments = []
    for i, (f, m) in enumerate(zip(frames, masks)):
        name = f"{i}.png"
        Image.fromarray(f).save(frames_root/name)
        names.append(name)
        # Store uncompressed column-major RLE, as accepted by the official API.
        flat = m.ravel(order="F"); counts=[]; bit=0; run=0
        for item in flat:
            if item == bit:
                run += 1
            else:
                counts.append(run); run=1; bit=int(item)
        counts.append(run)
        segments.append({"size": list(m.shape), "counts": counts})
    annotations = {"videos": [{"id": 1, "file_names": names, "width": 64, "height": 48}],
                   "categories": [{"id": 1, "name": "fixture-only"}],
                   "annotations": [{"id": 3, "video_id": 1, "category_id": 1, "segmentations": segments}]}
    annotation_file = tmp_path/"annotations.json"
    annotation_file.write_text(json.dumps(annotations))
    # Adapter test only: the source record explicitly marks procedural material.
    source = {"source": "YouTube-VIS", "snapshot": "ANALYTICAL-ADAPTER-TEST-NOT-REAL-DATA", "url": "https://example.invalid/fixture",
              "terms_url": "https://youtube-vos.org/dataset/term/", "terms_checked_on": "2026-10-31", "annotations_license": "fixture",
              "media_conditions": "procedural fixture only", "local_media_provenance": "test-generated", "usage_scope": "non-commercial-research"}
    source["synthetic"] = True
    config = {"dataset_version": "adapter-test", "max_clips": 2, "shard_size": 1, "seed": 0,
              "heldout_categories": [], "converter": {}, "quality": {"calibrated": False}}
    result = build_dataset(annotation_file, frames_root, source, config, tmp_path/"corpus", 3, contract)
    assert len(result["clips"]) == 1 and not result["report"]["D1_complete"]
    assert result["report"]["yield_full_clips"] == 1
    assert (tmp_path/"corpus"/"reviews.template.json").exists()
    _, _, box, meta = read_clip(tmp_path/"corpus", result["clips"][0], contract)
    assert len(meta["source_frame_indices"]) == 6 and len(meta["source_frame_sha256"]) == 6
    assert meta["label_scope"].startswith("subject only")
    assert meta["synthetic"] and result["clips"][0]["synthetic"]
    with pytest.raises(FileExistsError):
        build_dataset(annotation_file, frames_root, source, config, tmp_path/"corpus", 3, contract)
    with pytest.raises(ValueError, match="provenance"):
        validate_source({**source, "snapshot": "REPLACE_WITH_RELEASE"})

