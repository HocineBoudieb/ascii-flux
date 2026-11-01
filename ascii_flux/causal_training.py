"""Balanced real-data experiment for the native causal glyph transformer."""
import copy
from dataclasses import asdict,dataclass
import hashlib
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from .autoregressive import CausalConfig,CausalGlyphFlow,causal_loss
from .contract import TEXT_SPEC
from .masked_training import MaskedTrainConfig,subject_groups,token_batch
from .storage import atomic_file,write_json
from .training import Corpus,restore_rng,rng_state


@dataclass(frozen=True)
class CausalTrainConfig(MaskedTrainConfig):
    steps: int = 4000


def prepare(model,batch,validation=False):
    g,c,pg,pc,text = batch
    mode = torch.multinomial(torch.tensor([.65,.25,.10],device=g.device),len(g),replacement=True)
    pg,pc = pg.clone(),pc.clone()
    pg[mode == 0],pc[mode == 0] = model.mask_glyph,model.mask_color
    for i in (mode == 2).nonzero().flatten().tolist():
        others = ((text-text[i]).square().sum(-1) > 1e-8).nonzero().flatten()
        if len(others):
            j = int(others[torch.randint(len(others),(),device=g.device)])
            pg[i],pc[i] = g[j],c[j]
        else:
            mode[i] = 0
            pg[i],pc[i] = model.mask_glyph,model.mask_color
    if not validation:
        text = text*(torch.rand(len(g),1,device=g.device) >= model.config.text_dropout)
    return (g,c,pg,pc,text,mode)


@torch.inference_mode()
def validate(model,corpus,groups,config,device):
    saved = rng_state()
    model.eval()
    try:
        torch.manual_seed(config.seed+1)
        rng = random.Random(config.seed+1)
        values = []
        for _ in range(config.val_batches):
            batch = token_batch(corpus,groups,config,"val",device,rng)
            inputs = prepare(model,batch,True)
            with torch.autocast("cuda",dtype=torch.float16,enabled=str(device).startswith("cuda")):
                gl,cl = model(*inputs)
            values.append(float(causal_loss(model,gl,cl,batch[0],batch[1])))
        return float(np.mean(values))
    finally:
        restore_rng(saved)
        model.train()


def train_causal(manifest,embeddings,reviews,output,config=None,model_config=None,device="cuda",resume=None):
    config,model_config = config or CausalTrainConfig(),model_config or CausalConfig()
    if min(config.steps,config.batch_size,config.validate_every,config.val_batches) <= 0 or config.lr <= 0:
        raise ValueError("Positive training configuration required")
    corpus = Corpus(manifest,embeddings,reviews)
    groups = subject_groups(corpus,config.subjects)
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/"latest.pt").exists() and not resume:
        raise FileExistsError("Existing causal experiment; resume or choose a new run")
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
    model = CausalGlyphFlow(corpus.contract,model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.lr,weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda",enabled=str(device).startswith("cuda"),init_scale=128.)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    code = hashlib.sha256()
    for name in ("autoregressive.py","causal_training.py","masked_training.py","text.py"):
        code.update(Path(__file__).with_name(name).read_bytes())
    provenance = {"dataset_fingerprint":corpus.manifest["fingerprint"],"text_spec":TEXT_SPEC,
        "allow_fixture":False,"allow_unreviewed":False,"synthetic_encoder":False,
        "development_only":True,"recognition_validated":False,
        "scope":"real-data memorization diagnostic" if config.memorize_one_frame else "six-concept causal glyph experiment; recognition and motion unvalidated",
        "training_concepts":list(config.subjects),"source_conditions":corpus.manifest["source"],
        "review_scope":corpus.review_scope,"training_code_sha256":code.hexdigest(),"torch_version":str(torch.__version__)}
    start,best,best_step,best_model,history,skipped = 0,math.inf,0,None,[],0
    if resume:
        ck = torch.load(resume,map_location="cpu",weights_only=False)
        corpus.contract.require(ck["contract"])
        old,new = dict(ck["train_config"]),asdict(config)
        old.pop("steps");new.pop("steps")
        if ck["provenance"] != provenance or old != new or ck["model_config"] != asdict(model_config):
            raise ValueError("Causal resume provenance/config mismatch")
        model.load_state_dict(ck["model"]);ema.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["optimizer"]);scaler.load_state_dict(ck["scaler"])
        start,best,best_step,best_model,history,skipped = (ck[k] for k in ("step","best_val","best_step","best_model","history","skipped_amp_updates"))
        restore_rng(ck["rng"])
    started = time.monotonic()
    print({"parameters":sum(p.numel() for p in model.parameters()),"scope":provenance["scope"]},flush=True)
    for step in range(start,config.steps):
        batch = token_batch(corpus,groups,config,"train",device,random)
        inputs = prepare(model,batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.float16,enabled=str(device).startswith("cuda")):
            gl,cl = model(*inputs)
            loss = causal_loss(model,gl,cl,batch[0],batch[1])
        if not torch.isfinite(loss):raise FloatingPointError("Non-finite causal loss")
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=not scaler.is_enabled())
        scaler.step(optimizer);scaler.update()
        if math.isfinite(float(norm)):
            decay = min(.995,(1+step)/(10+step))
            with torch.no_grad():
                for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,1-decay)
        else:skipped += 1
        if (step+1)%25 == 0:
            write_json(output/"progress.json",{"step":step+1,"total_steps":config.steps,"loss":float(loss.detach()),
                "elapsed_seconds":time.monotonic()-started,"optimizer_updates":step+1-skipped,"recognition_validated":False})
        if (step+1)%config.validate_every == 0 or step+1 == config.steps:
            value = validate(ema,corpus,groups,config,device)
            if not math.isfinite(value):raise FloatingPointError("Non-finite causal validation")
            if value < best:
                best,best_step = value,step+1
                best_model = copy.deepcopy({k:v.detach().cpu() for k,v in ema.state_dict().items()})
            row = {"step":step+1,"train_ce":float(loss.detach()),"teacher_forced_val_ce":value,
                   "best_val_ce":best,"optimizer_updates":step+1-skipped}
            history.append(row)
            ck = {"format":"ascii-flux-v5-causal-1","contract":corpus.contract.to_dict(),"model_config":asdict(model_config),
                "train_config":asdict(config),"provenance":provenance,"model":model.state_dict(),"ema":ema.state_dict(),
                "optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),"step":step+1,"best_val":best,"best_step":best_step,
                "best_model":best_model,"rng":rng_state(),"history":history,"skipped_amp_updates":skipped}
            with atomic_file(output/"latest.pt") as stream:torch.save(ck,stream)
            with atomic_file(output/"best.pt") as stream:
                torch.save({k:ck[k] for k in ("format","contract","model_config","provenance")} | {"model":best_model,"step":best_step},stream)
            write_json(output/"history.json",history)
            print(row,flush=True)
    return model
