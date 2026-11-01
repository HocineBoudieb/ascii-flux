from dataclasses import replace
import json
import numpy as np
import pytest
from ascii_flux.catalogue import Catalogue
from ascii_flux.contract import Contract, digest
from ascii_flux.storage import write_json
from tests.fixtures import make_corpus


def test_catalogue_counts_repeats_and_idempotent_ingest(tmp_path):
    path, _ = make_corpus(tmp_path/"fixture")
    manifest = json.loads(path.read_text())
    contract = Contract.from_dict(manifest["contract"])
    with Catalogue(tmp_path/"db", contract, target_grids=24) as catalogue:
        assert catalogue.ingest(path)
        assert not catalogue.ingest(path)
        report = catalogue.report()
        assert report["clips"] == 4 and report["grid_slots"] == 24
        assert report["distinct_token_grids"] < report["grid_slots"]
        assert report["target_slots_reached"] and not report["target_distinct_grids_reached"]
        snapshot = catalogue.snapshot()
        from ascii_flux.training import load_manifest
        frozen, loaded = load_manifest(snapshot)
        assert len(frozen["clips"]) == 4 and loaded == contract
        assert catalogue.snapshot() == snapshot


def test_catalogue_rolls_back_video_split_leakage(tmp_path):
    path, _ = make_corpus(tmp_path/"fixture")
    manifest = json.loads(path.read_text())
    contract = Contract.from_dict(manifest["contract"])
    with Catalogue(tmp_path/"db", contract) as catalogue:
        catalogue.ingest(path)
        changed = dict(manifest)
        changed["clips"] = [dict(r, split="val") for r in manifest["clips"]]
        changed.pop("fingerprint")
        changed["fingerprint"] = digest(changed)
        write_json(path.parent/"leak.json", changed)
        with pytest.raises(ValueError, match="split leakage"):
            catalogue.ingest(path.parent/"leak.json")
        assert catalogue.report()["clips"] == 4
        assert catalogue.db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


def test_scale_resume_commits_each_job_once(tmp_path):
    from ascii_flux.scale import build_scale
    from tests.test_contract_and_factory import test_official_format_adapter_and_report
    # Reuse the explicitly tagged local adapter fixture, then give it two
    # source videos so a one-job budget must advance on the second call.
    test_official_format_adapter_and_report(tmp_path)
    annotations, frames = tmp_path/"annotations.json", tmp_path/"media"
    original = json.loads((tmp_path/"corpus/manifest.json").read_text())
    source, config = original["source"], original["config"]
    contract = Contract.from_dict(original["contract"])
    data = json.loads(annotations.read_text())
    data["videos"].append(dict(data["videos"][0], id=2))
    data["annotations"].append(dict(data["annotations"][0], id=4, video_id=2))
    write_json(annotations, data)
    config = dict(config, shard_size=1, target_grids=1000000, horizontal_flip=True)
    output = tmp_path/"scale"
    first = build_scale(annotations, frames, source, config, output, 3., workers=1, max_jobs=1, contract=contract)
    second = build_scale(annotations, frames, source, config, output, 3., workers=1, max_jobs=1, contract=contract)
    assert second["clips"] > first["clips"]
    assert second["clips"] == 4
    assert second["repeated_observation_slots"] == second["grid_slots"]//2
    assert second == build_scale(annotations, frames, source, config, output, 3., workers=1, contract=contract)
    # Repeated source images must keep the exact visually reviewed converter
    # result, despite the new adapter decoding each source image only once.
    from ascii_flux.factory import build_dataset
    from ascii_flux.storage import read_clip
    baseline = build_dataset(annotations, frames, source, dict(config, max_clips=1),
                             tmp_path/"d1-repeated", 1.5, contract)
    build_scale(annotations, frames, source, config, tmp_path/"scale-repeated", 1.5,
                workers=1, max_jobs=1, contract=contract)
    with Catalogue(tmp_path/"scale-repeated", contract) as catalogue:
        records = [json.loads(r[0]) for r in catalogue.db.execute("SELECT record FROM clips")]
    original = next(r for r in records if not r.get("augmentation"))
    flipped = next(r for r in records if r.get("augmentation"))
    a = read_clip(tmp_path/"d1-repeated", baseline["clips"][0], contract)
    b = read_clip(tmp_path/"scale-repeated", original, contract)
    f = read_clip(tmp_path/"scale-repeated", flipped, contract)
    for left, right in zip(a[:3], b[:3]):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(f[0], b[0][..., ::-1])
    np.testing.assert_array_equal(f[1], b[1][..., ::-1])
    np.testing.assert_array_equal(f[2], b[2])
