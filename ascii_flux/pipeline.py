"""Local D2 -> frozen corpus -> native NCA training, with resumable phases."""
from dataclasses import asdict
from contextlib import ExitStack
import json
import os
from pathlib import Path
import time
from .catalogue import Catalogue, writer_lock
from .storage import write_json


def run_million(project_root, data_root=None, workers=4):
    project = Path(project_root).resolve()
    data = Path(data_root).resolve() if data_root else project/"data"
    output = data/"v5/million-001"
    output.mkdir(parents=True, exist_ok=True)
    status_file = output/"pipeline-status.json"
    started = time.time()
    state = {"pid": os.getpid(), "started_unix": started, "target_grid_slots": 1_000_000}

    def status(phase, **extra):
        state.update(phase=phase, updated_unix=time.time(), **extra)
        write_json(status_file, state)
        print(json.dumps({"phase": phase, **extra}), flush=True)

    def read(name):
        return json.loads((project/name).read_text(encoding="utf-8"))

    # The long-lived pipeline lock is separate from short converter DB locks.
    with ExitStack() as stack:
        try:
            stack.enter_context(writer_lock(output/"pipeline"))
        except (BlockingIOError, PermissionError):
            existing = json.loads(status_file.read_text(encoding="utf-8")) if status_file.exists() else {"phase": "starting"}
            print(json.dumps({"already_running": True, "status": existing, "status_file": str(status_file)}), flush=True)
            return existing
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("Select the ASCII Flux - Radeon GPU kernel/interpreter; CPU fallback is disabled for this run")
            sample = data/"v5/d1-public-001/manifest.json"
            evidence = read("configs/d1_visual_review.json")
            if not sample.exists() or json.loads(sample.read_text(encoding="utf-8"))["fingerprint"] != evidence["sample_dataset_fingerprint"]:
                raise ValueError("Use the actual D1 sample reviewed by the user before launching scale conversion")
            status("acquisition", gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__))
            from .acquisition import load_prepared
            from .public_source import prepare_public_ytvis2019
            from .lvvis import prepare_lvvis
            ytvis = load_prepared(data, profile="ytvis-2019-public")
            if ytvis is None:
                ytvis = prepare_public_ytvis2019(data, read("configs/ytvis_source.json"), read("configs/ytvis2019_public.json"))
            lvvis = prepare_lvvis(data, read("configs/lvvis_public.json"))
            config = read("configs/d2.json")
            from .scale import build_scale
            for prepared in (ytvis, lvvis):
                status("conversion", source=prepared["source"]["source"])
                # A converter started interactively may still be finishing. Its
                # OS lock is released even after a crash; completed jobs are reused.
                while True:
                    try:
                        build_scale(prepared["annotations"], prepared["frames_root"], prepared["source"], config,
                                    output, prepared["source_fps"], workers=workers)
                        break
                    except (BlockingIOError, PermissionError) as error:
                        # Only lock acquisition errors have this precise marker.
                        # A real media/SQL error must be surfaced, not retried forever.
                        from .catalogue import writer_lock as lock
                        try:
                            with lock(output):
                                pass
                        except (BlockingIOError, PermissionError):
                            status("waiting_for_active_converter", source=prepared["source"]["source"])
                            time.sleep(20)
                            continue
                        raise error
            with Catalogue(output) as catalogue:
                report = catalogue.report()
                status("conversion_finished", grid_slots=report["grid_slots"], distinct_token_grids=report["distinct_token_grids"],
                       source_observations=report["distinct_source_object_observations"])
                # Never fill a shortfall by looping clips or silently relaxing quality.
                if report["grid_slots"] < config["target_grids"]:
                    raise ValueError("Quality-filtered source yield is below one million; add verified videos or explicitly revise the dataset plan")
                manifest = catalogue.snapshot("manifest-million-001.json")
            from .quality import bind_sample_review
            reviews = bind_sample_review(manifest, sample, output/"reviews-million-001.json",
                                         evidence["reviewer"], evidence["human_statement"])
            from .text import cache_embeddings
            embeddings = output/"text-million-001.npz"
            if not embeddings.exists():
                status("text_embeddings")
                previous = sample.with_name("text.real.npz")
                cache_embeddings(manifest, embeddings, "cuda", previous if previous.exists() else None)
            from .training import TrainConfig, train
            training = TrainConfig(**read("configs/train_local_8gb.json"))
            run = data/"runs/native-million-001"
            resume = run/"latest.pt"
            status("training", manifest=str(manifest), run=str(run), configuration=asdict(training))
            train(manifest, embeddings, run, training, reviews_path=reviews, device="cuda",
                  resume=resume if resume.exists() else None)
            status("training_finished", checkpoint=str(run/"best.pt"), recognition_validated=False,
                   elapsed_seconds=time.time()-started)
        except BaseException as error:
            status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   error=type(error).__name__+": "+str(error))
            raise
