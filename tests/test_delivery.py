import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_inference_never_imports_converter_or_vision_dependencies():
    subprocess.run([sys.executable, "-c", "import sys; from ascii_flux.runtime import FlowRuntime; from ascii_flux.training import load_inference; "
                    "from ascii_flux.inference import load_inference as load_native; "
                    "assert not any(m in sys.modules for m in ('ascii_flux.offline','ascii_flux.factory','cv2','PIL','transformers'))"],
                   cwd=ROOT, check=True, capture_output=True)


def test_autonomous_notebook_matches_delivered_sources():
    notebook = json.loads((ROOT/"ASCII_Flux_v5_D1_F1.ipynb").read_text(encoding="utf-8"))
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    for cell in cells.values():
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert not cell["outputs"] and cell["execution_count"] is None
    tree = ast.parse("".join(cells["bootstrap-v5"]["source"]))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and node.targets[0].id == "SOURCES")
    sources = ast.literal_eval(assignment.value)
    for name, content in sources.items():
        assert (ROOT/name).read_text(encoding="utf-8") == content
    checksum = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    assert checksum == notebook["metadata"]["ascii_flux"]["source_sha256"]
    assert "ascii_flux/player.html" in sources and "configs/train.json" in sources


def test_scale_notebook_contains_sources_and_gpu_pipeline():
    notebook = json.loads((ROOT/"ASCII_Flux_v5_D2_Million_GPU.ipynb").read_text(encoding="utf-8"))
    cells = {c["id"]: c for c in notebook["cells"]}
    for cell in cells.values():
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert not cell["outputs"] and cell["execution_count"] is None
    tree = ast.parse("".join(cells["bootstrap-d2"]["source"]))
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "SOURCES")
    sources = ast.literal_eval(assignment.value)
    assert "ascii_flux/pipeline.py" in sources and "configs/train_local_8gb.json" in sources
    for name, content in sources.items():
        assert (ROOT/name).read_text(encoding="utf-8") == content
    assert "run_million(WORK, DATA_ROOT" in "".join(cells["run-d2"]["source"])


def test_notebook_waits_for_acquisition_instead_of_raising_placeholder_error(tmp_path, capsys):
    notebook = json.loads((ROOT/"ASCII_Flux_v5_D1_F1.ipynb").read_text(encoding="utf-8"))
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    context = {"__name__": "__main__", "json": json, "Path": Path, "DATA_ROOT": tmp_path,
               "WORK": ROOT, "DATASET": tmp_path/"d1", "RUN": tmp_path/"run", "DEVICE": "cpu", "DOWNLOAD_DATA": False}
    # Opting out stops acquisition cleanly; subsequent cells must not attempt
    # to read a nonexistent corpus, download an encoder or train a model.
    context["input"] = lambda _: ""
    for name in ("acquire-v5", "convert-v5", "embed-v5", "train-v5", "measure-v5"):
        exec(compile(cells[name], name, "exec"), context)
    assert context["PREPARED"] is None and context["manifest"] is None
    assert not context["EMBEDDINGS_READY"]
    assert "en attente" in capsys.readouterr().out
    assert not (tmp_path/"d1").exists()


def test_review_cell_recovers_disk_manifest_when_memory_variable_is_none(tmp_path, monkeypatch, capsys):
    from tests.fixtures import make_corpus
    from ascii_flux.review import load_review_manifest
    display = pytest.importorskip("IPython.display")
    shown = []
    monkeypatch.setattr(display, "display", shown.append)
    manifest_path, _ = make_corpus(tmp_path/"d1")
    notebook = json.loads((ROOT/"ASCII_Flux_v5_D1_F1.ipynb").read_text(encoding="utf-8"))
    code = "".join(next(c for c in notebook["cells"] if c["id"] == "review-v5")["source"])
    context = {"DATASET": manifest_path.parent, "manifest": None}
    exec(compile(code, "review-v5", "exec"), context)
    assert context["manifest"]["fingerprint"] == load_review_manifest(manifest_path)["fingerprint"]
    assert context["REVIEW_HTML"].is_file() and shown
    assert "Lecteur prêt : 4 clips" in capsys.readouterr().out
