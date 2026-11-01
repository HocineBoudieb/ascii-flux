"""Training of the cascade glyph transformer on the teacher shape corpus.

Corpus records are tagged ``synthetic: teacher-t2i``. Words are split by hash:
``val`` and ``test_unseen`` words are never trained, so validation measures
drawing words the model has not been taught (an automatic proxy only).
"""
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
from .cascade import CascadeConfig, CascadeGlyphModel, cascade_loss
from .contract import canonical, digest
from .storage import atomic_file, write_json
from .structure import STRUCTURE
from .training import restore_rng, rng_state


@dataclass(frozen=True)
class CascadeTrainConfig:
    steps: int = 40000
    batch_size: int = 32
    lr: float = 4e-4
    warmup: int = 1000
    weight_decay: float = .05
    validate_every: int = 1000
    val_batches: int = 16
    flip: float = .5
    ema: float = .999
    seed: int = 20261031


def load_shape_corpus(root, shard_limit=None):
    """Read the first ``shard_limit`` shards, a fixed snapshot while generation continues."""
    root = Path(root)
    words, glyphs, colors = [], [], []
    shards = sorted(root.glob("shard-*.json"))[:shard_limit]
    fingerprint = hashlib.sha256()
    for sidecar in shards:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        data = sidecar.with_suffix(".npz").read_bytes()
        if hashlib.sha256(data).hexdigest() != meta["npz_sha256"] or meta["contract_hash"] != STRUCTURE.hash:
            raise ValueError(f"Invalid shape shard {sidecar.name}")
        fingerprint.update(meta["npz_sha256"].encode())
        import io
        with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
            offset = len(glyphs)
            g, c = arrays["glyphs"], arrays["colors"]
        glyphs.extend(g)
        colors.extend(c)
        for record in meta["words"]:
            if record["samples"]:
                words.append({"word": record["word"], "split": record["split"],
                              "indices": [offset+s["index"] for s in record["samples"]]})
    if not words:
        raise ValueError("Empty shape corpus")
    return words, np.asarray(glyphs, np.uint8), np.asarray(colors, np.uint8), fingerprint.hexdigest(), len(shards)


def phrases_for(word):
    return [word, f"a {word}"]


def cache_text_tokens(words, output, device="cuda", shards=None):
    from .text import TOKEN_SPEC, TextEncoder
    encoder = TextEncoder(device)
    names = sorted({w["word"] for w in words})
    texts = [p for name in names for p in phrases_for(name)]
    features = np.concatenate([encoder.encode_tokens(texts[i:i+128]) for i in range(0, len(texts), 128)])
    features = features.reshape(len(names), len(phrases_for("x")), *features.shape[1:])
    metadata = {"spec": TOKEN_SPEC, "words": len(names), "phrases": "word | a word", "corpus_shards": shards}
    with atomic_file(output) as stream:
        np.savez(stream, words=np.array(names), features=features, metadata=np.array(canonical(metadata)))
    return metadata


class ShapeData:
    def __init__(self, corpus_root, tokens_path, device):
        with np.load(tokens_path, allow_pickle=False) as cache:
            limit = json.loads(str(cache["metadata"])).get("corpus_shards")
        self.words, self.glyphs, self.colors, self.fingerprint, self.shards = load_shape_corpus(corpus_root, limit)
        with np.load(tokens_path, allow_pickle=False) as cache:
            index = {w: i for i, w in enumerate(cache["words"].tolist())}
            missing = {w["word"] for w in self.words}-set(index)
            if missing:
                raise ValueError(f"Text tokens missing for {len(missing)} words; rebuild the cache")
            self.features = torch.tensor(cache["features"])
            self.text_metadata = json.loads(str(cache["metadata"]))
        self.index = index
        self.device = device
        self.split = {s: [w for w in self.words if w["split"] == s] for s in ("train", "val", "test_unseen")}
        if not self.split["train"] or not self.split["val"]:
            raise ValueError("Need train and never-trained validation words")

    def batch(self, split, size, rng, flip=0.):
        chosen = [rng.choice(self.split[split]) for _ in range(size)]
        samples = [rng.choice(w["indices"]) for w in chosen]
        phrase = [rng.randrange(self.features.shape[1]) if split == "train" else 0 for _ in chosen]
        g = torch.tensor(self.glyphs[samples], dtype=torch.long, device=self.device)
        c = torch.tensor(self.colors[samples], dtype=torch.long, device=self.device)
        text = torch.stack([self.features[self.index[w["word"]], p] for w, p in zip(chosen, phrase)]).to(self.device)
        flips = torch.tensor([rng.random() < flip for _ in chosen], device=self.device)
        return g, c, text, flips


@torch.inference_mode()
def validate(model, data, config, device, split="val"):
    saved = rng_state()
    model.eval()
    try:
        rng = random.Random(config.seed+1)
        torch.manual_seed(config.seed+1)
        rows = {"with_text": [], "without_text": []}
        for i in range(config.val_batches):
            g, c, text, _ = data.batch(split, config.batch_size, rng)
            for key, context in (("with_text", model.text_context(text)), ("without_text", model.null_text)):
                # Same masks on every device for both conditions.
                torch.manual_seed(config.seed+100+i)
                with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
                    loss, parts = cascade_loss(model, g, c, context)
                rows[key].append(parts)
        mean = {k: {p: float(np.mean([r[p] for r in v])) for p in v[0]} for k, v in rows.items()}
        mean["silhouette_text_gain"] = mean["without_text"]["silhouette_ce"]-mean["with_text"]["silhouette_ce"]
        mean["total"] = sum(mean["with_text"].values())
        return mean
    finally:
        restore_rng(saved)
        model.train()


def train_cascade(corpus_root, tokens_path, output, config=None, model_config=None, device="cuda", resume=False):
    config, model_config = config or CascadeTrainConfig(), model_config or CascadeConfig()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"latest.pt").exists() and not resume:
        raise FileExistsError("Existing run; pass resume or choose a new output")
    data = ShapeData(corpus_root, tokens_path, device)
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    model = CascadeGlyphModel(STRUCTURE, model_config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim < 2 or "embedding" in name or "position" in name or "null" in name else decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": config.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.}], lr=config.lr, betas=(.9, .98))
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    code = hashlib.sha256()
    for name in ("cascade.py", "cascade_training.py", "structure.py"):
        code.update(Path(__file__).with_name(name).read_bytes())
    provenance = {"corpus_fingerprint": data.fingerprint, "corpus_shards": data.shards, "synthetic": "teacher-t2i",
                  "text": data.text_metadata, "train_words": len(data.split["train"]),
                  "never_trained_val_words": len(data.split["val"]), "training_code_sha256": code.hexdigest(),
                  "recognition_validated": False, "development_only": True,
                  "scope": "cascade silhouette->glyph transformer on teacher shapes; static shapes, no motion yet"}
    start, best, history = 0, math.inf, []
    if resume:
        checkpoint = torch.load(output/"latest.pt", map_location="cpu", weights_only=False)
        old, new = dict(checkpoint["train_config"]), asdict(config)
        old.pop("steps"); new.pop("steps")
        if checkpoint["provenance"] != provenance or old != new or checkpoint["model_config"] != asdict(model_config):
            raise ValueError("Resume provenance/config mismatch")
        model.load_state_dict(checkpoint["model"]); ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"]); scaler.load_state_dict(checkpoint["scaler"])
        start, best, history = checkpoint["step"], checkpoint["best"], checkpoint["history"]
        restore_rng(checkpoint["rng"])
    parameters = sum(p.numel() for p in model.parameters())
    print(json.dumps({"parameters": parameters, "train_words": len(data.split["train"]),
                      "val_words": len(data.split["val"]), "samples": len(data.glyphs)}), flush=True)
    started = time.monotonic()
    for step in range(start, config.steps):
        lr = config.lr*min(1, (step+1)/config.warmup)*.5*(1+math.cos(math.pi*min(1, step/config.steps)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        g, c, text, flips = data.batch("train", config.batch_size, random, config.flip)
        fg, fc = model.mirror(g, c)
        g, c = torch.where(flips[:, None, None], fg, g), torch.where(flips[:, None, None], fc, c)
        drop = torch.rand(len(g), device=device) < model_config.text_dropout
        with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
            loss, parts = cascade_loss(model, g, c, model.text_context(text, drop))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite cascade loss")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            rate = min(config.ema, (1+step)/(10+step))
            for a, b in zip(ema.parameters(), model.parameters()):
                a.lerp_(b, 1-rate)
        if (step+1) % 50 == 0:
            write_json(output/"progress.json", {"step": step+1, "total_steps": config.steps, "loss": float(loss),
                                                 **parts, "elapsed_seconds": time.monotonic()-started,
                                                 "recognition_validated": False})
        if (step+1) % config.validate_every == 0 or step+1 == config.steps:
            val = validate(ema, data, config, device)
            row = {"step": step+1, "train": parts, "val_never_trained_words": val}
            if step+1 == config.steps and data.split["test_unseen"]:
                row["test_unseen_final_report_only"] = validate(ema, data, config, device, "test_unseen")
            history.append(row)
            weights = {k: v.detach().cpu() for k, v in ema.state_dict().items()}
            if val["total"] < best:
                best = val["total"]
                with atomic_file(output/"best.pt") as stream:
                    torch.save({"format": "ascii-flux-v5-cascade-1", "contract": STRUCTURE.to_dict(),
                                "model_config": asdict(model_config), "provenance": provenance,
                                "model": weights, "step": step+1}, stream)
            with atomic_file(output/"latest.pt") as stream:
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict(), "step": step+1, "best": best, "history": history,
                            "rng": rng_state(), "train_config": asdict(config), "model_config": asdict(model_config),
                            "provenance": provenance}, stream)
            write_json(output/"history.json", history)
            print(json.dumps(row), flush=True)
    return model
