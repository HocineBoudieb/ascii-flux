import json
import shutil
import threading
import pytest
from ascii_flux.progress import dataset_progress, dataset_progress_snapshot
from ascii_flux.storage import write_json
from tests.fixtures import make_corpus


def test_progress_counts_only_saved_batches_then_reads_completed_manifest(tmp_path):
    fixture, _ = make_corpus(tmp_path/"fixture")
    output = tmp_path/"d1"
    staging = tmp_path/"d1.staging"
    staging.mkdir()
    shutil.copyfile(fixture.parent/"shard-00000.tar", staging/"shard-00000.tar")
    (staging/"shard-00001.tar.incomplete.tmp").write_bytes(b"unfinished")
    assert dataset_progress_snapshot(output) == {"complete": False, "saved": 4}
    value = json.loads(fixture.read_text())
    value["report"].update(machine_accepted=4, attempted_full_clips=7, rejected_records=3)
    write_json(staging/"manifest.json", value)
    staging.rename(output)
    assert dataset_progress_snapshot(output) == {"complete": True, "saved": 4, "attempted": 7, "rejections": 3}


def test_progress_heartbeat_and_cleanup_on_interruption(tmp_path):
    messages = []
    heartbeat = threading.Event()

    def emit(message):
        messages.append(message)
        if len(messages) >= 3:
            heartbeat.set()

    with pytest.raises(RuntimeError, match="fixture interruption"):
        with dataset_progress(tmp_path/"d1", 200, interval=.01, heartbeat=.02, emit=emit):
            assert heartbeat.wait(2)
            raise RuntimeError("fixture interruption")
    assert "interrompue" in messages[-1]
    assert not any("Terminé" in m for m in messages)
    assert not any(t.name == "ascii-flux-D1-progress" for t in threading.enumerate())
