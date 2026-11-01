"""Sample sheets from a cascade checkpoint; prompts are encoded live with SigLIP.

Each row shows the emergence (silhouette then glyphs) and two seeds. Words
are marked seen/never-trained from the checkpoint's corpus split.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image, ImageDraw, ImageFont
from ascii_flux.cascade import CascadeConfig, CascadeGlyphModel
from ascii_flux.contract import Contract
from ascii_flux.shape_corpus import split_of
from ascii_flux.storage import atomic_file, write_json
from ascii_flux.text import TextEncoder

ROOT = Path(__file__).resolve().parents[1]


def draw_grid(draw, font, contract, glyphs, colors, x, y, cw=6, ch=12):
    for r in range(contract.rows):
        for c in range(contract.cols):
            if glyphs[r, c]:
                draw.text((x+c*cw, y+r*ch), contract.glyphs[glyphs[r, c]], fill=contract.palette[colors[r, c]], font=font)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="data/runs/cascade-001/best.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--words", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--temperature", type=float)
    args = parser.parse_args()
    output = ROOT/args.output
    if (output/"scope.json").exists():
        raise FileExistsError("Published samples are immutable; choose a new --output")
    checkpoint = torch.load(ROOT/args.checkpoint, map_location="cpu", weights_only=True)
    contract = Contract.from_dict(checkpoint["contract"])
    config = CascadeConfig(**checkpoint["model_config"])
    config = replace(config, **{k: v for k, v in (("guidance", args.guidance), ("temperature", args.temperature)) if v is not None})
    model = CascadeGlyphModel(contract, config)
    model.load_state_dict(checkpoint["model"])
    model = model.to(args.device).eval()
    text = torch.tensor(TextEncoder(args.device).encode_tokens(args.words, config.text_tokens), device=args.device).float()
    runs = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        runs.append(model.sample(text, return_trace=True))
    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 11)
    pw, ph = 64*6+16, 32*12+30
    stages = [config.silhouette_steps//2-1, config.silhouette_steps-1, config.silhouette_steps+config.glyph_steps//2-1]
    columns = len(stages)+2
    sheet = Image.new("RGB", (pw*columns, ph*len(args.words)), "black")
    draw = ImageDraw.Draw(sheet)
    for i, word in enumerate(args.words):
        status = "jamais entraîné" if split_of(word) != "train" else "vocabulaire d'entraînement"
        for j, step in enumerate(stages):
            g, c = (t[i].cpu().numpy() for t in runs[0][2][step])
            draw.text((j*pw+8, i*ph+6), f"{word} — pas {step+1}", fill="white", font=font)
            draw_grid(draw, font, contract, g, c, j*pw+8, i*ph+24)
        for k, (g, c, _) in enumerate(runs):
            x = (len(stages)+k)*pw+8
            draw.text((x, i*ph+6), f"{word} — final, graine {k+1} ({status})", fill="white", font=font)
            draw_grid(draw, font, contract, g[i].cpu().numpy(), c[i].cpu().numpy(), x, i*ph+24)
    output.mkdir(parents=True, exist_ok=True)
    with atomic_file(output/"samples.png") as stream:
        sheet.save(stream, format="PNG")
    write_json(output/"scope.json", {"checkpoint": str(args.checkpoint), "step": checkpoint["step"], "words": args.words,
                                     "guidance": config.guidance, "temperature": config.temperature,
                                     "provenance": checkpoint["provenance"],
                                     "note": "author preview; not a forced-choice human recognition result"})
    print(output/"samples.png")


if __name__ == "__main__":
    main()
