import random
from types import SimpleNamespace
import numpy as np
import pytest
from ascii_flux.masked_training import MaskedTrainConfig, label_sampler, subject_groups
from ascii_flux.text import label_phrases


def test_label_phrases_turn_identifiers_into_typed_words():
    assert label_phrases("car_(automobile)") == ["car", "a car", "car moving", "car (automobile)"]
    assert label_phrases("Aerosol_Can")[0] == "aerosol can"
    assert label_phrases("cat") == ["cat", "a cat", "cat moving"]


def record(label, video, split):
    return {"normalized_label": label, "source": "S", "video_id": video, "split": split}


def corpus():
    groups = {"train": [record("cat", "v1", "train"), record("cat", "v2", "train"), record("cat", "v3", "train"),
                        record("cat", "v4", "train"), record("dog", "v5", "train")],
              "val": [record("cat", "v6", "val"), record("lynx", "v7", "val")],
              "test_seen": [], "test_unseen": [record("giraffe", "v8", "test_unseen"), record("cat", "v9", "test_unseen")]}
    return SimpleNamespace(groups=groups)


def test_open_vocabulary_separates_never_trained_labels():
    config = MaskedTrainConfig(subjects=())
    assert config.open_vocabulary
    groups = subject_groups(corpus(), config.subjects)
    assert set(groups["train"]) == {"cat", "dog"}
    assert set(groups["seen_val"]) == {"cat"}
    assert set(groups["unseen_val"]) == {"lynx"}
    # A label already trained is not counted as unseen even in the reserved split.
    assert set(groups["unseen_test"]) == {"giraffe"}


def test_label_balance_tempers_frequent_labels():
    groups = subject_groups(corpus(), ())
    names, weights = label_sampler(groups, MaskedTrainConfig(subjects=(), label_balance=.5))["train"]
    assert names == ["cat", "dog"] and np.allclose(weights, [2., 1.])
    names, weights = label_sampler(groups, MaskedTrainConfig(subjects=(), label_balance=0.))["train"]
    assert weights == [1., 1.]
    draws = random.Random(0).choices(*label_sampler(groups, MaskedTrainConfig(subjects=()))["train"], k=3000)
    assert 0.6 < draws.count("cat")/3000 < 0.73


def test_single_subject_is_rejected():
    with pytest.raises(ValueError):
        MaskedTrainConfig(subjects=("cat",))
    with pytest.raises(ValueError):
        MaskedTrainConfig(subjects=(), label_balance=2)
