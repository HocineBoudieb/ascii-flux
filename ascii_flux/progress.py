"""Observe D1 output without modifying conversion or its checkpoint identity."""
from contextlib import contextmanager
import json
from pathlib import Path
import tarfile
import threading
import time


def dataset_progress_snapshot(output):
    """Count only atomically published clips; ignore incomplete temporary files."""
    output = Path(output)
    manifest = output/"manifest.json"
    try:
        if manifest.is_file():
            value = json.loads(manifest.read_text(encoding="utf-8"))
            report = value["report"]
            return {"complete": True, "saved": report.get("machine_accepted", len(value["clips"])),
                    "attempted": report.get("attempted_full_clips"), "rejections": report.get("rejected_records", 0)}
        stages = [p for p in output.parent.glob(output.name+".*") if p.is_dir()]
        if not stages:
            return {"complete": False, "saved": 0}
        stage = max(stages, key=lambda p: p.stat().st_mtime)
        count = 0
        for shard in stage.glob("shard-*.tar"):
            with tarfile.open(shard, "r") as archive:
                count += sum(member.isfile() and member.name.endswith(".npz") for member in archive)
        return {"complete": False, "saved": count}
    except (OSError, ValueError, KeyError, tarfile.TarError):
        # The staging directory may be renamed to the final path while observed.
        return {"complete": False, "saved": None}


def _emit(message):
    print(message, flush=True)


@contextmanager
def dataset_progress(output, max_candidates, interval=2., heartbeat=20., emit=None):
    """Notebook progress from saved batches, plus a heartbeat during each batch.

    No model, converter function or input data is patched. In particular this
    observer can be added without invalidating an existing converter's identity.
    """
    if max_candidates <= 0 or interval <= 0 or heartbeat <= 0:
        raise ValueError("Invalid progress limits")
    emit = emit or _emit
    stopped = threading.Event()
    started = time.monotonic()
    previous_count, last_message = -1, started

    def update(force=False):
        nonlocal previous_count, last_message
        status = dataset_progress_snapshot(output)
        now = time.monotonic()
        if status["saved"] is None:
            return
        if force or status["saved"] != previous_count or now-last_message >= heartbeat:
            seconds = int(now-started)
            duration = f"{seconds//60} min {seconds%60:02d} s"
            if status["complete"]:
                message = f"[D1] Terminé : {status['saved']} clips sauvegardés"
                if status["attempted"] is not None:
                    message += f", {status['attempted']} candidats traités"
                message += f", {status['rejections']} rejets. Durée : {duration}."
            else:
                message = (f"[D1] Conversion en cours : {status['saved']} clips sauvegardés "
                           f"(jusqu’à {max_candidates} candidats). Durée : {duration}.")
            emit(message)
            previous_count, last_message = status["saved"], now

    def observe():
        while not stopped.wait(interval):
            update()

    emit("[D1] Conversion CPU. Le compteur avance à chaque lot sauvegardé ; un lot peut prendre plusieurs minutes.")
    update(force=True)
    worker = threading.Thread(target=observe, name="ascii-flux-D1-progress", daemon=True)
    worker.start()
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        stopped.set()
        worker.join(timeout=2)
        if succeeded:
            update(force=True)
        else:
            emit("[D1] Conversion interrompue ou en erreur ; aucun résultat incomplet n'est annoncé comme terminé.")
