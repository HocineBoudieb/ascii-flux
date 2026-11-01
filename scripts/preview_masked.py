"""Offline native-token samples (masked or causal). Cache is evaluation-only."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from ascii_flux.storage import atomic_file,write_json
from ascii_flux.inference import load_inference

ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",default="data/runs/masked-six-001/best.pt")
    parser.add_argument("--output",default="outputs/masked-six-preview-001")
    parser.add_argument("--emergence-steps",type=int,default=8)
    parser.add_argument("--guidance",type=float,default=2.)
    parser.add_argument("--temperature",type=float,default=.7)
    parser.add_argument("--training-state",action="store_true",
                        help="Read our own full training checkpoint and preview its current raw weights")
    parser.add_argument("--labels",nargs="+",help="Dataset labels to draw; default: every training concept")
    args = parser.parse_args()
    output = ROOT/args.output
    output.mkdir(parents=True,exist_ok=True)
    if (output/'scope.json').exists():
        raise FileExistsError('Published samples are immutable; choose a new --output directory')
    # Snapshot the atomically published checkpoint while training may continue.
    snapshot = (ROOT/args.checkpoint).read_bytes()
    with atomic_file(output/"checkpoint.pt") as stream:stream.write(snapshot)
    checkpoint = torch.load(output/"checkpoint.pt",map_location="cpu",weights_only=not args.training_state)
    checkpoint_step = checkpoint["step"]
    if args.training_state:
        # The full training artifact contains RNG/optimizer objects; the
        # inference artifact deliberately keeps only weights and their contract.
        with atomic_file(output/"checkpoint.pt") as stream:
            torch.save({k:checkpoint[k] for k in ("format","contract","model_config","provenance","model","step")},stream)
    model,provenance = load_inference(output/"checkpoint.pt","cuda",allow_development=True)
    if hasattr(model,"patch_count"):
        cycle = (model.patch_count+model.config.patches_per_step-1)//model.config.patches_per_step
        stages = (cycle,cycle*2)
        model.config = replace(model.config,guidance=args.guidance,temperature=args.temperature)
    else:
        stages = (args.emergence_steps,args.emergence_steps+56)
        model.config = replace(model.config,emergence_steps=args.emergence_steps,
                               guidance=args.guidance,temperature=args.temperature)
    labels = args.labels or provenance["training_concepts"]
    with np.load(ROOT/"data/v5/million-001/text-million-001.npz",allow_pickle=False) as cache:
        mapping = dict(zip(cache["labels"].tolist(),cache["embeddings"]))
    text = torch.tensor(np.stack([mapping[label] for label in labels]),device="cuda")
    saved = {}
    for seed in (42,123):
        torch.manual_seed(seed)
        state = model.initial_state(len(labels))
        for step in range(1,stages[-1]+1):
            state,_,_ = model.step(state,text)
            if step in stages:
                g,c = model.tokens(state)
                saved[(seed,step)] = (g.cpu().numpy().astype(np.uint8),c.cpu().numpy().astype(np.uint8))
    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf",11)
    cw,ch,pw,ph = 6,12,400,420
    canvas = Image.new("RGB",(pw*3,ph*len(labels)),"black")
    draw = ImageDraw.Draw(canvas)
    for row,label in enumerate(labels):
        for col,key in enumerate(((42,stages[0]),(123,stages[0]),(42,stages[1]))):
            g,c = saved[key]
            x,y = col*pw+8,row*ph+8
            draw.text((x,y),f"{label} - seed {key[0]} / {key[1]} pas",fill="white",font=font)
            for r in range(model.contract.rows):
                for cc in range(model.contract.cols):
                    if g[row,r,cc]:
                        draw.text((x+cc*cw,y+23+r*ch),model.contract.glyphs[g[row,r,cc]],
                                  fill=model.contract.palette[c[row,r,cc]],font=font)
    with atomic_file(output/"samples.png") as stream:canvas.save(stream,format="PNG")
    # Compact exact tokens permit later side-by-side review without image losses.
    with atomic_file(output/"tokens.npz") as stream:
        np.savez_compressed(stream,**{f"glyphs_{s}_{t}":v[0] for (s,t),v in saved.items()},
                            **{f"colors_{s}_{t}":v[1] for (s,t),v in saved.items()})
    write_json(output/"scope.json",{"checkpoint":args.checkpoint,"checkpoint_step":checkpoint_step,
               "checkpoint_sha256":hashlib.sha256(snapshot).hexdigest(),
               "inference_snapshot_sha256":hashlib.sha256((output/"checkpoint.pt").read_bytes()).hexdigest(),
               "training_state":args.training_state,
               "sampling":model.config_dict(),"provenance":provenance,
               "scope":"generated token samples; not human recognition or a validated model", "labels":labels})
    print(str(output/"samples.png"),flush=True)


if __name__ == "__main__":main()
