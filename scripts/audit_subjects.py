"""Reproducible gallery of real training tokens, not model generations."""
from collections import Counter
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageDraw,ImageFont
from ascii_flux.contract import Contract
from ascii_flux.storage import read_clip,write_json


ROOT = Path(__file__).resolve().parents[1]
LABELS = ("cat","dog","horse","bird","fish","person")


def main():
    root = ROOT/"data/v5/million-001"
    output = ROOT/"outputs/subject-audit-001"
    output.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((root/"manifest-million-001.json").read_text(encoding="utf-8"))
    contract = Contract.from_dict(manifest["contract"])
    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf",11)
    rng = random.Random(42)
    evidence = {"scope":"real training examples; not generations or human recognition scores",
                "dataset_fingerprint":manifest["fingerprint"],"subjects":{}}
    for label in LABELS:
        records = [r for r in manifest["clips"] if r["split"] == "train"
                   and r["normalized_label"] == label and not r.get("augmentation")]
        by_video = {}
        for record in records:
            by_video.setdefault((record["source"],record["video_id"]),[]).append(record)
        videos = rng.sample(list(by_video),min(8,len(by_video)))
        selected = [rng.choice(by_video[v]) for v in videos]
        canvas = Image.new("RGB",(1600,840),"black")
        draw = ImageDraw.Draw(canvas)
        for index,record in enumerate(selected):
            g,c,_,meta = read_clip(root,record,contract)
            x,y = index%4*400+8,index//4*420+8
            draw.text((x,y),f"{label} / {record['source']} / {record['video_id']}",font=font,fill="white")
            for r in range(contract.rows):
                for col in range(contract.cols):
                    if g[12,r,col]:
                        draw.text((x+col*6,y+23+r*12),contract.glyphs[g[12,r,col]],
                                  fill=contract.palette[c[12,r,col]],font=font)
        canvas.save(output/f"{label}.png")
        evidence["subjects"][label] = {"original_clips":len(records),"source_videos":len(by_video),
            "sources":dict(Counter(r["source"] for r in records)),
            "gallery_clip_ids":[r["clip_id"] for r in selected],"frame_index":12}
    write_json(output/"audit.json",evidence)
    print(str(output),flush=True)


if __name__ == "__main__":main()
