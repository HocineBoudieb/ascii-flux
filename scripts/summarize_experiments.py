"""Publish the failed recognition audit, keeping numerical and visual claims apart."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from ascii_flux.contract import DEFAULT
from ascii_flux.storage import atomic_file,write_json


ROOT = Path(__file__).resolve().parents[1]


def sampling_metrics(folder):
    scope = json.loads((folder/"scope.json").read_text(encoding="utf-8"))
    result = {"checkpoint_step":scope.get("checkpoint_step"),"sampling":scope.get("sampling"),
              "source_checkpoint_sha256":scope.get("checkpoint_sha256"),"labels":scope["labels"]}
    palette = np.array(DEFAULT.palette)
    with np.load(folder/"tokens.npz",allow_pickle=False) as tokens:
        stages = sorted(int(k.rsplit('_',1)[1]) for k in tokens.files if k.startswith('glyphs_42_'))
        result["stages"] = {}
        for step in stages:
            g,c = tokens[f'glyphs_42_{step}'],tokens[f'colors_42_{step}']
            active = g != 0
            visible = active & palette[np.minimum(c,63)].any(-1)
            result["stages"][str(step)] = {"nonempty_cells":active.sum((1,2)).tolist(),
                                         "nonblack_nonempty_cells":visible.sum((1,2)).tolist()}
        a,b = tokens[f'glyphs_42_{stages[0]}'],tokens[f'glyphs_42_{stages[-1]}']
        result["glyph_cells_changed_between_stages"] = (a != b).sum((1,2)).tolist()
        result["scope"] = "token counts, not recognition or convincing motion"
    return result


def main():
    output = ROOT/'outputs/recognition-audit-001'
    corpus = json.loads((ROOT/'outputs/subject-audit-001/audit.json').read_text(encoding='utf-8'))
    masked = ROOT/'outputs/masked-six-at-6000'
    causal = ROOT/'outputs/causal-six-final'
    nca_ck = torch.load(ROOT/'data/runs/native-million-001/best.pt',map_location='cpu',weights_only=True)
    code = hashlib.sha256()
    for name in ('training.py','model.py','text.py','contract.py'):
        code.update(name.encode());code.update((ROOT/'ascii_flux'/name).read_bytes())
    histories = {name:json.loads((ROOT/'data/runs'/name/'history.json').read_text(encoding='utf-8'))
                 for name in ('masked-six-001','causal-six-001')}
    report = {"status":"not_ready_for_recognizable_generation","human_recognition_accuracy":None,
        "visual_review_scope":"assistant qualitative inspection; not a human recognition study",
        "dataset_fingerprint":corpus['dataset_fingerprint'],"real_training_coverage":corpus['subjects'],
        "nca":{"best_step":nca_ck['step'],"training_steps":20000,
               "training_identity_preserved":code.hexdigest() == nca_ck['provenance']['training_code_sha256'],
               "visual_result":"failed; generic shapes and degradation across long rollout"},
        "masked":{"training_steps":histories['masked-six-001'][-1]['step'],
                  "visual_result":"failed; average shapes remain ambiguous", "samples":sampling_metrics(masked)},
        "causal":{"training_steps":histories['causal-six-001'][-1]['step'],
                  "best_teacher_forced_validation_step":min(histories['causal-six-001'],key=lambda r:r['teacher_forced_val_ce'])['step'],
                  "visual_result":"failed; inconsistent shapes and subject loss in self-conditioned sampling",
                  "samples":sampling_metrics(causal)},
        "capacity_test":{"scope":"memorization of six real training frames only",
                         "generalization_validated":False,"motion_validated":False},
        "interpretation":{"supported":["rare concepts have few independent videos", "balanced sampling alone did not fix generation",
            "teacher-forced validation does not establish generation quality"],
            "hypotheses":["ambiguous category-only captions and partial subjects limit the prior",
                          "training on real previous frames does not prepare the model for its own malformed outputs"]},
        "deployment":{"default_checkpoint_changed":False,"F1_validated":False,"F2_validated":False}}
    discrete = ROOT/'outputs/discrete-six-best'
    if (discrete/'tokens.npz').exists():
        history = json.loads((ROOT/'data/runs/discrete-six-001/history.json').read_text(encoding='utf-8'))
        report['discrete'] = {'training_steps':history[-1]['step'],
            'visual_result':'failed; user recognized no subject on the six-generation sheet',
            'samples':sampling_metrics(discrete)}
        report['discrete']['human_review'] = json.loads((ROOT/'configs/generation_visual_review.json').read_text(encoding='utf-8'))
        review = report['discrete']['human_review']
        for field,filename in (('inference_snapshot_sha256','checkpoint.pt'),('tokens_sha256','tokens.npz'),('gallery_sha256','concepts.png')):
            if hashlib.sha256((discrete/filename).read_bytes()).hexdigest() != review[field]:
                raise ValueError('Human generation review belongs to different evidence: '+filename)
        report['D2_horse_sample_human_review'] = json.loads((ROOT/'configs/d2_visual_review.json').read_text(encoding='utf-8'))
        with np.load(discrete/'tokens.npz',allow_pickle=False) as f:
            dg,dc = f['glyphs_42_32'],f['colors_42_32']
        font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf',11)
        sheet = Image.new('RGB',(1200,840),'black');painter = ImageDraw.Draw(sheet)
        for index,label in enumerate(report['discrete']['samples']['labels']):
            x,y = index%3*400+8,index//3*420+8
            painter.text((x,y),label+' - generation native / 32 pas',font=font,fill='white')
            for r in range(DEFAULT.rows):
                for k in range(DEFAULT.cols):
                    if dg[index,r,k]:painter.text((x+k*6,y+23+r*12),DEFAULT.glyphs[dg[index,r,k]],font=font,
                                                  fill=DEFAULT.palette[dc[index,r,k]])
        # Keep the exact, hash-pinned sheet the user evaluated.
    write_json(output/'conclusion.json',report)
    # Native-token render only, using a real training reference and generations.
    with np.load(output/'samples.npz',allow_pickle=False) as f:
        truth_g,truth_c = f['target_glyphs'],f['target_colors']
        ng,nc = f['glyphs_8'],f['colors_8']
    with np.load(masked/'tokens.npz',allow_pickle=False) as f:mg,mc = f['glyphs_42_8'],f['colors_42_8']
    with np.load(causal/'tokens.npz',allow_pickle=False) as f:cg,cc = f['glyphs_42_32'],f['colors_42_32']
    font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf',11)
    canvas = Image.new('RGB',(1600,840),'black');draw = ImageDraw.Draw(canvas)
    titles = ('reference reelle (train)','NCA / generation','masque / generation','causal / generation')
    for row,index in enumerate((0,2)):
        for col,(g,c) in enumerate(((truth_g,truth_c),(ng,nc),(mg,mc),(cg,cc))):
            x,y = col*400+8,row*420+8
            draw.text((x,y),f"{('cat','horse')[row]} - {titles[col]}",font=font,fill='white')
            for r in range(DEFAULT.rows):
                for k in range(DEFAULT.cols):
                    if g[index,r,k]:draw.text((x+k*6,y+23+r*12),DEFAULT.glyphs[g[index,r,k]],font=font,
                                             fill=DEFAULT.palette[c[index,r,k]])
    with atomic_file(output/'experiments.png') as stream:canvas.save(stream,format='PNG')
    print(json.dumps({'report':str(output/'conclusion.json'),'training_identity_preserved':report['nca']['training_identity_preserved'],
                      'status':report['status']}),flush=True)


if __name__ == '__main__':main()
