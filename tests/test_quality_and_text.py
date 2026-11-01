from dataclasses import replace
import json
import numpy as np
import pytest
from ascii_flux.contract import TEXT_SPEC, canonical, digest
from ascii_flux.quality import bind_sample_review, calibrated_view
from ascii_flux.storage import write_json
from ascii_flux.text import cache_embeddings
from ascii_flux.training import Corpus
from tests.fixtures import make_corpus


def test_sample_review_is_bound_to_policy_and_keeps_fixture_gate(tmp_path):
    path, cache = make_corpus(tmp_path/"fixture")
    original = json.loads(path.read_text())
    quality = {"calibrated": True, "calibration_sample_fingerprint": original["fingerprint"]}
    view = calibrated_view(path, path.with_name("view.json"), quality)
    review = bind_sample_review(view, path, path.with_name("sample-review.json"), "test reviewer", "Analytical test only")
    vectors = np.ones((2, 768), np.float32)
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    # Explicit test material remains synthetic even after a sample review.
    with np.load(cache, allow_pickle=False) as f:
        meta = json.loads(str(f["metadata"]))
    meta["dataset_fingerprint"] = json.loads(view.read_text())["fingerprint"]
    np.savez_compressed(cache, labels=np.array(["fixture a", "fixture b"]), embeddings=vectors, metadata=canonical(meta))
    with pytest.raises(ValueError, match="Synthetic"):
        Corpus(view, cache, review)
    corpus = Corpus(view, cache, review, allow_fixture=True)
    assert corpus.review_scope == "converter-sample"
    bad = json.loads(review.read_text());bad["quality_policy"] = {"calibrated": True}
    write_json(review, bad)
    with pytest.raises(ValueError, match="sample review"):
        Corpus(view, cache, review, allow_fixture=True)


def test_genuine_text_cache_rebinds_without_loading_an_encoder(tmp_path, monkeypatch):
    path, cache = make_corpus(tmp_path/"fixture")
    with np.load(cache, allow_pickle=False) as f:
        labels, vectors = f["labels"], f["embeddings"]
        meta = json.loads(str(f["metadata"]))
    with pytest.raises(ValueError, match="genuine"):
        cache_embeddings(path, tmp_path/"new.npz", existing_cache=cache)
    # Stub genuine marker is a unit fixture only, kept inside tmp_path.
    meta["synthetic_encoder"] = False
    np.savez_compressed(cache, labels=labels, embeddings=vectors, metadata=canonical(meta))
    class ForbiddenEncoder:
        def __init__(self, *args):
            raise AssertionError("No model should load when every label is already cached")
    monkeypatch.setattr("ascii_flux.text.TextEncoder", ForbiddenEncoder)
    cache_embeddings(path, tmp_path/"new.npz", existing_cache=cache)
    with np.load(tmp_path/"new.npz", allow_pickle=False) as f:
        np.testing.assert_array_equal(f["embeddings"], vectors)
        assert json.loads(str(f["metadata"]))["vision_tower_loaded"] is False
