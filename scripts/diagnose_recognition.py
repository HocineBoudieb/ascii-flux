"""Compare genuine ASCII targets and native predictions under controlled noise.

The matching scores are analytical token-space diagnostics, not human recognition.
"""
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ascii_flux.model import token_loss
from ascii_flux.storage import read_clip, write_json
from ascii_flux.training import load_inference

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = ROOT/"data/v5/million-001"
    output = ROOT/"outputs/recognition-audit-001"
    output.mkdir(exist_ok=True)
    manifest = json.loads((data/"manifest-million-001.json").read_text(encoding="utf-8"))
    model, _ = load_inference(ROOT/"data/runs/native-million-001/best.pt", "cuda")
    labels = ["cat", "dog", "horse", "bird", "fish", "person"]
    with np.load(data/"text-million-001.npz", allow_pickle=False) as f:
        vectors = dict(zip(f["labels"].tolist(), f["embeddings"]))
    text = torch.tensor(np.stack([vectors[k] for k in labels]), device="cuda")
    selected = []
    for label in labels:
        choices = [r for r in manifest["clips"] if r["split"] == "train" and r["normalized_label"] == label
                   and not r.get("augmentation")]
        choices = [r for r in choices if r["source"] == "YouTube-VIS"] or choices
        # Select a substantial visible subject; no target is used during sampling.
        selected.append(min(choices, key=lambda r: abs(r["quality"]["occupancy"]-.2)))
    clips = [read_clip(data, r, model.contract) for r in selected]
    tg = torch.tensor(np.stack([c[0][12] for c in clips]).astype(np.int64), device="cuda")
    tc = torch.tensor(np.stack([c[1][12] for c in clips]).astype(np.int64), device="cuda")
    saved, norms, scores = {}, {}, {}
    torch.manual_seed(20261031)
    with torch.inference_mode():
        state = model.initial_state(len(labels))
        for step in range(1, 601):
            shape = (1, 1, model.contract.rows, model.contract.cols)
            noise = torch.randn(shape, device="cuda").expand(len(labels), -1, -1, -1)
            mask = (torch.rand(shape, device="cuda") < model.config.update_probability).float().expand(len(labels), -1, -1, -1)
            state, gl, cl = model(state, text, noise, mask)
            if step in (8, 16, 48, 240, 600):
                g, c = model.tokens(state)
                saved[step] = (g.cpu().numpy().astype(np.uint8), c.cpu().numpy().astype(np.uint8))
                norms[str(step)] = float(state.abs().max())
                # Each row: target frame. Each column: text condition generating
                # an independent grid from exactly the same noise trajectory.
                costs = [[float(token_loss(gl[j:j+1], cl[j:j+1], tg[i:i+1], tc[i:i+1], 4.))
                          for j in range(len(labels))] for i in range(len(labels))]
                costs = np.array(costs)
                foreground = g != 0
                iou = ((foreground & (tg != 0)).sum((1, 2)) / (foreground | (tg != 0)).sum((1, 2)).clamp_min(1)).cpu().tolist()
                scores[str(step)] = {"target_vs_prompt_ce": costs.tolist(),
                                     "matched_ce_mean": float(np.diag(costs).mean()),
                                     "wrong_text_ce_mean": float(costs[~np.eye(len(labels), dtype=bool)].mean()),
                                     "nearest_text_matches": sum(costs.argmin(1) == np.arange(len(labels))).item(),
                                     "foreground_iou_against_selected_training_frames": iou,
                                     "scope": "aligned token comparison on training examples; not semantic recognition"}
    cells = []
    for i, label in enumerate(labels):
        cells.append({"label": label, "train_clips_with_augmentation": sum(r["split"] == "train" and r["normalized_label"] == label for r in manifest["clips"]),
                      "target_clip_id": selected[i]["clip_id"]})
    write_json(output/"diagnostics.json", {"labels": labels, "state_abs_max": norms, "scores": scores,
               "sample_counts": cells, "human_recognition": "not measured"})
    np.savez_compressed(output/"samples.npz", **{f"glyphs_{step}": a[0] for step, a in saved.items()},
                        **{f"colors_{step}": a[1] for step, a in saved.items()}, target_glyphs=tg.cpu().numpy(), target_colors=tc.cpu().numpy())
    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 11)
    cw, ch, panelw, panelh = 6, 12, 400, 420
    canvas = Image.new("RGB", (panelw*3, panelh*3), "black")
    draw = ImageDraw.Draw(canvas)
    for row, index in enumerate((0, 2, 5)):
        panels = [(tg[index].cpu().numpy(), tc[index].cpu().numpy(), "donnee reelle"),
                  (saved[8][0][index], saved[8][1][index], "modele / 8 pas"),
                  (saved[600][0][index], saved[600][1][index], "modele / 600 pas")]
        for col, (glyphs, colors, title) in enumerate(panels):
            x, y = col*panelw+8, row*panelh+8
            draw.text((x, y), labels[index]+" - "+title, fill="white", font=font)
            for r in range(model.contract.rows):
                for c in range(model.contract.cols):
                    if glyphs[r,c]:
                        draw.text((x+c*cw, y+23+r*ch), model.contract.glyphs[glyphs[r,c]],
                                  fill=model.contract.palette[colors[r,c]], font=font)
    canvas.save(output/"comparison.png")
    print(json.dumps({"state_abs_max": norms, "scores": {k:{a:b for a,b in v.items() if a != "target_vs_prompt_ce"} for k,v in scores.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
