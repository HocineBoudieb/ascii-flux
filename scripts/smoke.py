"""Analytical end-to-end smoke; explicitly not D1 or a semantic model."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.fixtures import make_corpus
from ascii_flux.contract import Contract
from ascii_flux.evaluation import export_onnx, measure_rollout
from ascii_flux.model import ModelConfig
from ascii_flux.review import review_page
from ascii_flux.storage import read_clip, write_json
from ascii_flux.training import TrainConfig, load_inference, train


def main():
    torch.set_num_threads(1)
    run = ROOT/"outputs"/"analytical-smoke"/time.strftime("%Y%m%d-%H%M%S")
    manifest_path, embeddings_path = make_corpus(run/"corpus")
    config = TrainConfig(steps=2, batch_size=1, emergence_steps=2, rollout_frames=2, transition_steps=2,
                         validate_every=1, val_batches=1, long_every=2, long_steps=500, long_grad_steps=2)
    train(manifest_path, embeddings_path, run/"training", config, ModelConfig(state_channels=8, width=16),
          allow_unreviewed=True, allow_fixture=True)
    model, provenance = load_inference(run/"training/best.pt", allow_development=True)
    with np.load(embeddings_path, allow_pickle=False) as cache:
        vectors = cache["embeddings"]
    result = measure_rollout(model, vectors[0], steps=500, switch_embedding=vectors[1])
    result["scope"] = "ANALYTICAL FIXTURE ONLY: procedural blocks and random text vectors; no semantic evidence"
    result["provenance"] = provenance
    result["D1_complete"] = result["F1_complete"] = False
    try:
        result["onnx_max_abs_errors"] = export_onnx(model, run/"step.onnx")
    except ImportError:
        result["onnx"] = "not run: optional export dependencies unavailable"
    review_page(manifest_path, run/"review.html")
    manifest = json.loads(manifest_path.read_text())
    contract = Contract.from_dict(manifest["contract"])
    glyphs, colors, _, _ = read_clip(manifest_path.parent, manifest["clips"][0], contract)
    # Direct display of stored character tokens, not an inference converter.
    from PIL import Image, ImageDraw, ImageFont
    cell_w, cell_h = 20, round(20*contract.cell_height_width)
    preview = Image.new("RGB", (contract.cols*cell_w, contract.rows*cell_h), "black")
    draw = ImageDraw.Draw(preview)
    font_path = Path("C:/Windows/Fonts/consola.ttf")
    if not font_path.exists():
        font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    font = ImageFont.truetype(str(font_path), cell_h-4) if font_path.exists() else ImageFont.load_default()
    for y in range(contract.rows):
        for x in range(contract.cols):
            glyph = int(glyphs[0, y, x])
            if glyph:
                draw.text((x*cell_w, y*cell_h), contract.glyphs[glyph], font=font,
                          fill=tuple(contract.palette[int(colors[0, y, x])]))
    preview.save(run/"fixture-preview.png")
    write_json(run/"diagnostics.json", result)
    write_json(ROOT/"outputs/analytical-smoke/latest.json", {"path": str(run), "scope": result["scope"]})
    print(json.dumps({"path": str(run), "finite": result["finite"], "D1_complete": False, "F1_complete": False}, indent=2))


if __name__ == "__main__":
    main()
