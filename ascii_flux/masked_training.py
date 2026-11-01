"""Controlled masked-token experiment on the existing real ASCII corpus."""
from collections import defaultdict
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from .contract import TEXT_SPEC
from .masked import MaskedConfig, MaskedGlyphFlow, masked_loss
from .storage import atomic_file, write_json
from .training import Corpus, restore_rng, rng_state


@dataclass(frozen=True)
class MaskedTrainConfig:
    steps: int = 3000
    batch_size: int = 12
    lr: float = .0003
    subjects: tuple = ("cat", "dog", "horse", "bird", "fish", "person")
    validate_every: int = 250
    val_batches: int = 8
    seed: int = 20261031
    memorize_one_frame: bool = False
    # Exponent of the label-frequency sampling: 1 follows the corpus, 0 is
    # uniform over labels. Uniform over-samples labels with a single clip.
    label_balance: float = .5

    def __post_init__(self):
        object.__setattr__(self,"subjects",tuple(self.subjects))
        if len(set(self.subjects)) != len(self.subjects) or len(self.subjects) == 1:
            raise ValueError("Choose at least two distinct real-data subjects, or none for every training label")
        if not 0 <= self.label_balance <= 1:
            raise ValueError("label_balance must lie in [0, 1]")

    @property
    def open_vocabulary(self):
        return not self.subjects


def _by_label(records):
    labels = defaultdict(lambda: defaultdict(list))
    for record in records:
        labels[record["normalized_label"]][(record["source"], record["video_id"])].append(record)
    return {label: dict(videos) for label,videos in labels.items()}


def subject_groups(corpus, subjects):
    """Train/val groups; with no subjects, every label plus never-trained labels.

    ``unseen_val`` holds validation videos whose label never occurs in training:
    the only direct measure of drawing a word the model was not taught.
    ``unseen_test`` comes from the reserved test partition and is reported
    once at the end, never used to choose a checkpoint.
    """
    if not subjects:
        result = {split: _by_label(corpus.groups[split]) for split in ("train", "val")}
        trained = set(result["train"])
        result["seen_val"] = {l: v for l,v in result["val"].items() if l in trained}
        result["unseen_val"] = {l: v for l,v in result["val"].items() if l not in trained}
        result["unseen_test"] = {l: v for l,v in _by_label(corpus.groups["test_unseen"]).items() if l not in trained}
        if not result["seen_val"]:
            raise ValueError("No validation video shares a label with training")
        return result
    result = {}
    for split in ("train", "val"):
        labels = _by_label(r for r in corpus.groups[split] if r["normalized_label"] in subjects)
        if set(subjects)-set(labels):
            raise ValueError(f"Missing {split} subjects: {sorted(set(subjects)-set(labels))}")
        result[split] = labels
    result["seen_val"] = result["val"]
    return result


def label_sampler(groups, config):
    """Per-group label lists and weights; frequency is counted in source videos."""
    samplers = {}
    for name, labels in groups.items():
        names = sorted(labels)
        if name == "train":
            weights = [len(labels[l])**config.label_balance for l in names]
        else:
            weights = [1.]*len(names)
        samplers[name] = (names, weights)
    return samplers


def token_batch(corpus, groups, config, split, device, rng, samplers=None, phrases=None):
    if samplers and split in samplers:
        names, weights = samplers[split]
        labels = rng.choices(names, weights, k=config.batch_size)
    else:
        labels = [rng.choice(config.subjects) for _ in range(config.batch_size)]
    targets_g, targets_c, previous_g, previous_c, vectors = [], [], [], [], []
    for label in labels:
        videos = groups[split][label]
        if config.memorize_one_frame and split == "train":
            candidates = [r for rs in videos.values() for r in rs if not r.get("augmentation")]
            record = min(candidates, key=lambda r: abs(r["quality"]["occupancy"]-.2))
            t, delta = 12, 1
        else:
            record = rng.choice(videos[rng.choice(list(videos))])
            t, delta = rng.randrange(3, corpus.contract.frames), rng.choice((1,2,3))
        g,c,_,_ = corpus.read(record)
        targets_g.append(g[t]);targets_c.append(c[t])
        previous_g.append(g[t-delta]);previous_c.append(c[t-delta])
        if phrases is not None:
            # Training sees every phrasing; evaluation uses the canonical one,
            # the closest to what a person types.
            options = phrases[label]
            vectors.append(options[rng.randrange(len(options))] if split == "train" else options[0])
        else:
            vectors.append(corpus.embeddings[label])
    return (torch.tensor(np.stack(targets_g), dtype=torch.long, device=device),
            torch.tensor(np.stack(targets_c), dtype=torch.long, device=device),
            torch.tensor(np.stack(previous_g), dtype=torch.long, device=device),
            torch.tensor(np.stack(previous_c), dtype=torch.long, device=device),
            torch.tensor(np.stack(vectors), device=device))


def corrupt(model, batch, validation=False):
    g,c,pg,pc,text = batch
    count = len(g)
    mode = torch.multinomial(torch.tensor([.65,.25,.10], device=g.device), count, replacement=True)
    # Broad mask ratios include the entirely unknown initial state.
    ratio = torch.rand(count, device=g.device)*.9+.05
    ratio = torch.where(torch.rand(count, device=g.device) < .20, 1., ratio)
    ratio = torch.where(mode != 0, torch.maximum(ratio, ratio.new_tensor(.7)), ratio)
    mask = torch.rand_like(g.float()) < ratio[:,None,None]
    xg = g.masked_fill(mask, model.mask_glyph)
    xc = c.masked_fill(mask, model.mask_color)
    pg,pc = pg.clone(),pc.clone()
    pg[mode == 0], pc[mode == 0] = model.mask_glyph, model.mask_color
    # A different real subject supplies transition context; no generated media.
    pg[mode == 2],pc[mode == 2] = g.roll(1,0)[mode == 2],c.roll(1,0)[mode == 2]
    # Slight context corruption exposes the temporal model to imperfect tokens.
    context_mask = (torch.rand_like(g.float()) < .04) & (mode[:,None,None] == 1)
    pg[context_mask],pc[context_mask] = model.mask_glyph,model.mask_color
    if not validation:
        text = text * (torch.rand(count,1,device=g.device) >= model.config.text_dropout)
    return (xg,xc,pg,pc,text,mask.float().mean((1,2)),mode),mask


@torch.inference_mode()
def validate(model, corpus, groups, config, device, split="seen_val", samplers=None, phrases=None, with_gain=False):
    """Masked CE on fixed draws. With ``with_gain``, also the CE without text.

    The text gain (CE without text minus CE with text) is positive only if the
    prompt actually informs the grid. On labels never trained, it measures
    transfer through the encoder; it is an automatic proxy, not recognition.
    """
    saved = rng_state()
    model.eval()
    try:
        rng = random.Random(config.seed+1)
        torch.manual_seed(config.seed+1)
        values, unconditioned = [], []
        for _ in range(config.val_batches):
            batch = token_batch(corpus, groups, config, split, device, rng, samplers, phrases)
            inputs, mask = corrupt(model, batch, validation=True)
            with torch.autocast("cuda",dtype=torch.float16,enabled=str(device).startswith("cuda")):
                gl,cl = model(*inputs)
                values.append(float(masked_loss(gl,cl,batch[0],batch[1],mask)))
                if with_gain:
                    blind = inputs[:4]+(torch.zeros_like(inputs[4]),)+inputs[5:]
                    gl,cl = model(*blind)
                    unconditioned.append(float(masked_loss(gl,cl,batch[0],batch[1],mask)))
        if not with_gain:
            return float(np.mean(values))
        return {"ce": float(np.mean(values)), "ce_without_text": float(np.mean(unconditioned)),
                "text_gain": float(np.mean(unconditioned)-np.mean(values)), "labels": len(groups[split])}
    finally:
        restore_rng(saved)
        model.train()


def train_masked(manifest_path, embeddings_path, reviews_path, output, config=None, model_config=None, device="cuda", resume=None,
                 phrases_path=None):
    config, model_config = config or MaskedTrainConfig(), model_config or MaskedConfig()
    if min(config.steps,config.batch_size,config.validate_every,config.val_batches) <= 0 or config.lr <= 0:
        raise ValueError("Positive training configuration required")
    corpus = Corpus(manifest_path, embeddings_path, reviews_path)
    groups = subject_groups(corpus, config.subjects)
    samplers = label_sampler(groups, config) if config.open_vocabulary else None
    phrases, phrase_metadata = (None, None)
    if phrases_path:
        from .text import load_phrase_embeddings
        phrases, phrase_metadata = load_phrase_embeddings(phrases_path, corpus.manifest["fingerprint"])
        if set(corpus.embeddings)-set(phrases):
            raise ValueError("Phrase cache misses dataset labels")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"latest.pt").exists() and not resume:
        raise FileExistsError("Existing masked experiment; resume or choose a new run")
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
    model = MaskedGlyphFlow(corpus.contract,model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.lr,weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"), init_scale=256.)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    code = hashlib.sha256()
    for name in ("masked.py","masked_training.py","text.py"):
        code.update(Path(__file__).with_name(name).read_bytes())
    provenance = {"dataset_fingerprint":corpus.manifest["fingerprint"],"text_spec":TEXT_SPEC,
                  "allow_fixture":False,"allow_unreviewed":False,"synthetic_encoder":False,
                  "recognition_validated":False,"development_only":True,
                  "scope":("real-data memorization diagnostic" if config.memorize_one_frame else
                           "open-vocabulary masked glyph experiment over every training label; recognition and motion unvalidated"
                           if config.open_vocabulary else "six-concept masked glyph experiment; recognition and motion unvalidated"),
                  "training_concepts":sorted(groups["train"]) if config.open_vocabulary else list(config.subjects),
                  "never_trained_validation_labels":sorted(groups.get("unseen_val",{})),
                  "phrases":phrase_metadata,"source_conditions":corpus.manifest["source"],
                  "review_scope":corpus.review_scope,"training_code_sha256":code.hexdigest(),
                  "torch_version":str(torch.__version__)}
    best,best_model,best_step,start,history = math.inf,None,0,0,[]
    if resume:
        checkpoint = torch.load(resume,map_location="cpu",weights_only=False)
        corpus.contract.require(checkpoint["contract"])
        old,new = dict(checkpoint["train_config"]),asdict(config)
        old.pop("steps");new.pop("steps")
        if checkpoint["provenance"] != provenance or old != new or checkpoint["model_config"] != asdict(model_config):
            raise ValueError("Masked resume provenance/config mismatch")
        model.load_state_dict(checkpoint["model"]);ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"]);scaler.load_state_dict(checkpoint["scaler"])
        start,best,best_model,history = checkpoint["step"],checkpoint["best_val"],checkpoint["best_model"],checkpoint["history"]
        best_step = checkpoint["best_step"]
        restore_rng(checkpoint["rng"])
    started = time.monotonic()
    print(json.dumps({"parameters":sum(p.numel() for p in model.parameters()),"training_labels":len(groups["train"]),
                      "never_trained_val_labels":len(groups.get("unseen_val",{})),"scope":provenance["scope"]}),flush=True)
    for step in range(start,config.steps):
        batch = token_batch(corpus,groups,config,"train",device,random,samplers,phrases)
        inputs,mask = corrupt(model,batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.float16,enabled=str(device).startswith("cuda")):
            gl,cl = model(*inputs)
            loss = masked_loss(gl,cl,batch[0],batch[1],mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite masked-token loss")
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=not scaler.is_enabled())
        scaler.step(optimizer);scaler.update()
        if math.isfinite(float(norm)):
            decay = min(.999,(1+step)/(10+step))
            with torch.no_grad():
                for a,b in zip(ema.parameters(),model.parameters()):
                    a.lerp_(b,1-decay)
        if (step+1) % 25 == 0:
            write_json(output/"progress.json",{"step":step+1,"total_steps":config.steps,"loss":float(loss.detach()),
                         "elapsed_seconds":time.monotonic()-started,"recognition_validated":False})
        if (step+1) % config.validate_every == 0 or step+1 == config.steps:
            if config.open_vocabulary:
                seen = validate(ema,corpus,groups,config,device,"seen_val",samplers,phrases,with_gain=True)
                value = seen["ce"]
            else:
                value = validate(ema,corpus,groups,config,device)
            if not math.isfinite(value):
                raise FloatingPointError("Non-finite masked validation")
            # The checkpoint is chosen on labels seen in training only.
            if value < best:
                best,best_model = value,copy.deepcopy({k:v.detach().cpu() for k,v in ema.state_dict().items()})
                best_step = step+1
            row = {"step":step+1,"masked_train_ce":float(loss.detach()),"masked_val_ce":value,"best_val_ce":best}
            if config.open_vocabulary:
                row["seen_val"] = seen
                if groups["unseen_val"]:
                    row["unseen_val"] = validate(ema,corpus,groups,config,device,"unseen_val",samplers,phrases,with_gain=True)
                if step+1 == config.steps and groups["unseen_test"]:
                    row["unseen_test_final_report_only"] = validate(ema,corpus,groups,config,device,"unseen_test",samplers,phrases,with_gain=True)
            history.append(row)
            checkpoint = {"format":"ascii-flux-v5-masked-1","contract":corpus.contract.to_dict(),"model_config":asdict(model_config),
                          "train_config":asdict(config),"provenance":provenance,"model":model.state_dict(),"ema":ema.state_dict(),
                          "optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),"step":step+1,"best_val":best,"best_step":best_step,
                          "best_model":best_model,"rng":rng_state(),"history":history}
            with atomic_file(output/"latest.pt") as stream:torch.save(checkpoint,stream)
            with atomic_file(output/"best.pt") as stream:
                torch.save({k:checkpoint[k] for k in ("format","contract","model_config","provenance")} | {"model":best_model,"step":best_step},stream)
            write_json(output/"history.json",history)
            print(json.dumps(row),flush=True)
    return model
