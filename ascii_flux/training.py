"""Research baseline: emergence, dynamics, concept switch and long rollouts.

Validation is token CE, never a claim of human recognition or semantic success.
"""
from dataclasses import asdict, dataclass
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from .contract import Contract, TEXT_SPEC, digest
from .model import GlyphNCA, ModelConfig, token_loss
from .storage import atomic_file, read_clip, write_json


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 10000
    batch_size: int = 2
    lr: float = 0.0003
    emergence_steps: int = 8
    rollout_frames: int = 4
    steps_per_frame: int = 2
    transition_steps: int = 8
    long_every: int = 100
    long_steps: int = 512
    long_grad_steps: int = 8
    validate_every: int = 100
    val_batches: int = 8
    teacher_decay_steps: int = 5000
    seed: int = 20261031
    precision: str = "float32"
    gradient_accumulation: int = 1
    foreground_weight: float = 1.

    def __post_init__(self):
        values = asdict(self)
        if any(v <= 0 for k, v in values.items() if k not in ("seed", "precision")):
            raise ValueError("Training counts and learning rate must be positive")
        if self.long_steps < 500 or self.long_grad_steps > self.long_steps:
            raise ValueError("Long rollouts require >=500 steps and a bounded gradient tail")
        if self.precision not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unknown training precision")


def load_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    content = dict(manifest)
    fingerprint = content.pop("fingerprint")
    if digest(content) != fingerprint:
        raise ValueError("Manifest fingerprint mismatch")
    contract = Contract.from_dict(manifest["contract"])
    if contract.hash != manifest["contract_hash"]:
        raise ValueError("Representation fingerprint mismatch")
    videos = {}
    heldout = set(manifest["config"]["heldout_categories"])
    for record in manifest["clips"]:
        key = (record["source"], record["video_id"])
        if key in videos and videos[key] != record["split"]:
            raise ValueError("Source video leaks between partitions")
        videos[key] = record["split"]
        if record["category"] in heldout and record["split"] != "test_unseen":
            raise ValueError("Reserved category leaks into development")
    # Hash every shard once when opening a run; corrupted tokens must not train silently.
    from .factory import file_sha256
    for filename, expected in manifest["shards"].items():
        shard = (path.parent / filename).resolve()
        if not shard.is_relative_to(path.parent.resolve()) or file_sha256(shard) != expected:
            raise ValueError(f"Invalid shard: {filename}")
    return manifest, contract


class Corpus:
    def __init__(self, manifest_path, embeddings_path, reviews_path=None, allow_unreviewed=False, allow_fixture=False):
        self.root = Path(manifest_path).parent
        self.manifest, self.contract = load_manifest(manifest_path)
        self.allow_unreviewed, self.allow_fixture = allow_unreviewed, allow_fixture
        if not self.manifest["config"]["quality"].get("calibrated") and not allow_unreviewed:
            raise ValueError("Calibrate converter quality thresholds after D1 review, or explicitly use --allow-unreviewed for development")
        decisions = {}
        if reviews_path:
            reviews = json.loads(Path(reviews_path).read_text(encoding="utf-8"))
            if reviews["dataset_fingerprint"] != self.manifest["fingerprint"] or reviews["converter_hash"] != self.manifest["converter"]["code_sha256"]:
                raise ValueError("Reviews belong to another dataset/converter")
            if not reviews.get("reviewer", "").strip():
                raise ValueError("Reviews need an identified reviewer")
            decisions = reviews.get("decisions", {})
            if reviews.get("scope") == "converter-sample":
                if (reviews.get("converter_identity") != self.manifest["converter"] or
                        reviews.get("quality_policy") != self.manifest["config"]["quality"] or
                        not reviews.get("human_statement", "").strip() or not reviews.get("sample_dataset_fingerprint")):
                    raise ValueError("Invalid converter sample review evidence")
                sample_reviewed = True
            else:
                sample_reviewed = False
        else:
            sample_reviewed = False
        self.review_scope = "converter-sample" if sample_reviewed else "per-clip" if reviews_path else "unreviewed"
        with np.load(embeddings_path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadata"]))
            if metadata["dataset_fingerprint"] != self.manifest["fingerprint"]:
                raise ValueError("Text cache belongs to another dataset")
            if metadata.get("synthetic_encoder") and not allow_fixture:
                raise ValueError("Synthetic text fixtures are not semantic embeddings")
            if metadata["text_spec"] != TEXT_SPEC:
                raise ValueError("Text encoder contract mismatch")
            vectors = cache["embeddings"]
            if vectors.shape != (len(cache["labels"]), TEXT_SPEC["dimension"]) or not np.isfinite(vectors).all():
                raise ValueError("Invalid embedding cache")
            self.embeddings = dict(zip(cache["labels"].tolist(), vectors))
            self.embedding_metadata = metadata
        self.groups = {name: [] for name in ("train", "val", "test_seen", "test_unseen")}
        for record in self.manifest["clips"]:
            decision = decisions.get(record["clip_id"], {}).get("decision", "pending")
            if decision == "reject" or (decision != "accept" and not allow_unreviewed and not sample_reviewed):
                continue
            if record.get("synthetic") and not allow_fixture:
                continue
            if record["normalized_label"] not in self.embeddings:
                raise ValueError("Missing label embedding")
            self.groups[record["split"]].append(record)
        if not self.groups["train"] or not self.groups["val"]:
            raise ValueError("Need separate nonempty train and validation videos after review")
        self._cache = OrderedDict()
        self._train_labels = defaultdict(list)
        for record in self.groups["train"]:
            self._train_labels[record["normalized_label"]].append(record)

    def read(self, record):
        key = (record["shard"], record["clip_id"])
        if key not in self._cache:
            self._cache[key] = read_clip(self.root, record, self.contract)
            if len(self._cache) > 64:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]

    def other_records(self, records, rng=None):
        rng = rng or random
        if len(self._train_labels) < 2:
            return []
        return [rng.choice(self._train_labels[rng.choice([label for label in self._train_labels
                 if label != record["normalized_label"]])]) for record in records]

    def batch(self, split, batch_size, device, rng=None):
        rng = rng or random
        records = [rng.choice(self.groups[split]) for _ in range(batch_size)]
        clips = [self.read(r) for r in records]
        glyphs = torch.as_tensor(np.stack([c[0] for c in clips]).astype(np.int64), device=device)
        colors = torch.as_tensor(np.stack([c[1] for c in clips]).astype(np.int64), device=device)
        text = torch.as_tensor(np.stack([self.embeddings[r["normalized_label"]] for r in records]), device=device)
        return glyphs, colors, text, records


def rollout(model, state, text, steps):
    for _ in range(steps):
        state, glyph_logits, color_logits = model.step(state, text)
    return state, glyph_logits, color_logits


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def precision_context(config, device):
    if config.precision == "float32":
        return nullcontext()
    if torch.device(device).type != "cuda":
        raise ValueError("Mixed precision training requires a tested CUDA/ROCm GPU")
    dtype = torch.float16 if config.precision == "float16" else torch.bfloat16
    return torch.autocast("cuda", dtype=dtype)


@torch.no_grad()
def validate(model, corpus, config, device):
    # Preserve training RNG so validation cadence does not alter optimization.
    saved = rng_state()
    model.eval()
    try:
        random.seed(config.seed + 1)
        torch.manual_seed(config.seed + 1)
        values = []
        for _ in range(config.val_batches):
            glyphs, colors, text, _ = corpus.batch("val", config.batch_size, device)
            state, gl, cl = rollout(model, model.initial_state(config.batch_size), text, config.emergence_steps)
            values.append(float(token_loss(gl, cl, glyphs[:, 0], colors[:, 0], config.foreground_weight)))
            for t in range(1, min(config.rollout_frames+1, corpus.contract.frames)):
                state, gl, cl = rollout(model, state, text, config.steps_per_frame)
                values.append(float(token_loss(gl, cl, glyphs[:, t], colors[:, t], config.foreground_weight)))
        return float(np.mean(values))
    finally:
        restore_rng(saved)
        model.train()

def training_losses(model, corpus, config, model_config, step, device):
    glyphs, colors, text, records = corpus.batch("train", config.batch_size, device)
    conditioned = text * (torch.rand(config.batch_size, 1, device=device) >= model_config.text_dropout)
    t = random.randrange(corpus.contract.frames-config.rollout_frames)
    state, gl, cl = rollout(model, model.initial_state(config.batch_size, noisy=bool(step % 2)), conditioned, config.emergence_steps)
    emergence = token_loss(gl, cl, glyphs[:, t], colors[:, t], config.foreground_weight)
    # Teacher forcing gradually gives way to recurrent predictions.
    teacher_probability = max(.1, 1-step/config.teacher_decay_steps)
    dynamics = emergence*0
    for k in range(1, config.rollout_frames+1):
        if random.random() < teacher_probability:
            state = model.lift(glyphs[:, t+k-1], colors[:, t+k-1])
        state, gl, cl = rollout(model, state, conditioned, config.steps_per_frame)
        dynamics = dynamics + token_loss(gl, cl, glyphs[:, t+k], colors[:, t+k], config.foreground_weight) / config.rollout_frames
    # Select a genuinely different label for B; skip if the corpus has one label.
    chosen = corpus.other_records(records)
    transition = emergence*0
    if chosen:
        clips = [corpus.read(r) for r in chosen]
        b_glyphs = torch.as_tensor(np.stack([c[0][0] for c in clips]).astype(np.int64), device=device)
        b_colors = torch.as_tensor(np.stack([c[1][0] for c in clips]).astype(np.int64), device=device)
        b_text = torch.as_tensor(np.stack([corpus.embeddings[r["normalized_label"]] for r in chosen]), device=device)
        b_text = b_text * (torch.rand(config.batch_size, 1, device=device) >= model_config.text_dropout)
        _, gl, cl = rollout(model, state.detach(), b_text, config.transition_steps)
        transition = token_loss(gl, cl, b_glyphs, b_colors, config.foreground_weight)
    long_loss = emergence*0
    if (step+1) % config.long_every == 0:
        with torch.no_grad():
            long_state, _, _ = rollout(model, model.initial_state(config.batch_size, noisy=True), conditioned,
                                       config.long_steps-config.long_grad_steps)
        _, gl, cl = rollout(model, long_state, conditioned, config.long_grad_steps)
        long_loss = token_loss(gl, cl, glyphs[:, t], colors[:, t], config.foreground_weight)
    loss = emergence + dynamics + transition + .25*long_loss
    return loss, teacher_probability, {"emergence_ce": emergence, "dynamics_ce": dynamics, "transition_ce": transition, "long_ce": long_loss}


def train(manifest_path, embeddings_path, output, config=None, model_config=None, reviews_path=None,
          device="cpu", resume=None, allow_unreviewed=False, allow_fixture=False):
    config, model_config = config or TrainConfig(), model_config or ModelConfig()
    corpus = Corpus(manifest_path, embeddings_path, reviews_path, allow_unreviewed, allow_fixture)
    if config.rollout_frames >= corpus.contract.frames:
        raise ValueError("rollout_frames must be shorter than a clip")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"latest.pt").exists() and not resume:
        raise FileExistsError("Existing run: supply --resume or choose another output")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = GlyphNCA(corpus.contract, model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    precision_context(config, device)  # fail before starting an incompatible run
    scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "float16", init_scale=256.)
    skipped_updates = 0
    best, best_model, best_step, start = math.inf, None, 0, 0
    reviews_hash = None
    if reviews_path:
        from .factory import file_sha256
        reviews_hash = file_sha256(reviews_path)
    provenance = {"dataset_fingerprint": corpus.manifest["fingerprint"], "text_spec": TEXT_SPEC,
                  "reviews_sha256": reviews_hash, "review_scope": corpus.review_scope, "allow_unreviewed": allow_unreviewed,
                  "allow_fixture": allow_fixture, "synthetic_encoder": corpus.embedding_metadata.get("synthetic_encoder", False),
                  "recognition_validated": False, "training_device": str(device), "torch_version": str(torch.__version__)}
    provenance["source_conditions"] = corpus.manifest["source"]
    provenance["converter"] = corpus.manifest["converter"]
    code_hash = hashlib.sha256()
    for name in ("training.py", "model.py", "text.py", "contract.py"):
        code_hash.update(name.encode())
        code_hash.update(Path(__file__).with_name(name).read_bytes())
    provenance["training_code_sha256"] = code_hash.hexdigest()
    history = []
    if resume:
        # Full resume contains optimizer + Python/NumPy RNG; only load our trusted local checkpoints.
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        corpus.contract.require(checkpoint["contract"])
        if checkpoint["model_config"] != asdict(model_config) or checkpoint["provenance"] != provenance:
            raise ValueError("Resume provenance/model mismatch")
        old_cfg, new_cfg = dict(checkpoint["train_config"]), asdict(config)
        old_cfg.pop("steps"); new_cfg.pop("steps")
        if old_cfg != new_cfg:
            raise ValueError("Resume training configuration mismatch (only total steps may change)")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        skipped_updates = checkpoint.get("skipped_amp_updates", 0)
        best, best_model, start = checkpoint["best_val"], checkpoint["best_model"], checkpoint["step"]
        best_step = checkpoint.get("best_step", start)
        if config.steps < start:
            raise ValueError("Total steps cannot precede the saved training step")
        history = checkpoint["history"]
        restore_rng(checkpoint["rng"])
    for step in range(start, config.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        metrics = defaultdict(float)
        total_loss = 0.
        for micro in range(config.gradient_accumulation):
            with precision_context(config, device):
                loss, teacher_probability, components = training_losses(model, corpus, config, model_config, step, device)
                scaled_loss = loss / config.gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; latest atomic checkpoint remains available")
            scaler.scale(scaled_loss).backward()
            total_loss += float(loss.detach()) / config.gradient_accumulation
            for key, value in components.items():
                metrics[key] += float(value.detach()) / config.gradient_accumulation
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=not scaler.is_enabled())
        if not torch.isfinite(norm):
            # GradScaler detected these infinities during unscale, skips the
            # optimizer update, and backs off its scale. Never count it as an
            # optimizer update or poison the weights with infinite gradients.
            skipped_updates += 1
        scaler.step(optimizer)
        scaler.update()
        if (step+1) % config.validate_every == 0 or step+1 == config.steps:
            with precision_context(config, device):
                value = validate(model, corpus, config, device)
            if not math.isfinite(value):
                raise FloatingPointError("Non-finite validation; keep the preceding atomic checkpoint")
            if value < best:
                best, best_model = value, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                best_step = step+1
            row = {"step": step+1, "train_ce": total_loss, "val_token_ce": value,
                   "best_val_token_ce": best, "teacher_probability": teacher_probability,
                   "optimizer_updates": step+1-skipped_updates, "skipped_amp_updates": skipped_updates, **metrics}
            history.append(row)
            checkpoint = {"format": "ascii-flux-v5-1", "contract": corpus.contract.to_dict(),
                          "model_config": asdict(model_config), "train_config": asdict(config), "provenance": provenance,
                          "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "best_model": best_model,
                          "best_val": best, "best_step": best_step, "step": step+1, "rng": rng_state(), "history": history,
                          "skipped_amp_updates": skipped_updates}
            with atomic_file(output/"latest.pt") as stream:
                torch.save(checkpoint, stream)
            # Tensor-only inference artifact: safe weights_only loading, no optimizer/RNG pickle.
            with atomic_file(output/"best.pt") as stream:
                torch.save({k: checkpoint[k] for k in ("format", "contract", "model_config", "provenance", "step")} |
                           {"model": best_model, "step": best_step}, stream)
            write_json(output/"history.json", history)
            print(json.dumps(row), flush=True)
    return model


def load_inference(path, device="cpu", allow_development=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format"] != "ascii-flux-v5-1":
        raise ValueError("Unknown checkpoint format")
    provenance = checkpoint["provenance"]
    if provenance["text_spec"] != TEXT_SPEC:
        raise ValueError("Text representation mismatch")
    if (provenance["allow_fixture"] or provenance["allow_unreviewed"]) and not allow_development:
        raise ValueError("Development checkpoint: explicitly enable development mode")
    model = GlyphNCA(Contract.from_dict(checkpoint["contract"]), ModelConfig(**checkpoint["model_config"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model, provenance
