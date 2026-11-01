"""Cascade glyph transformer on the teacher shape corpus (synthetic: teacher-t2i)."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ascii_flux.cascade import CascadeConfig
from ascii_flux.cascade_training import CascadeTrainConfig, cache_text_tokens, load_shape_corpus, train_cascade

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/v5/shapes-001")
    parser.add_argument("--output", default="data/runs/cascade-001")
    parser.add_argument("--tokens", help="Text token cache; rebuilt for the current corpus when omitted")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=CascadeTrainConfig.steps)
    parser.add_argument("--batch-size", type=int, default=CascadeTrainConfig.batch_size)
    parser.add_argument("--validate-every", type=int, default=CascadeTrainConfig.validate_every)
    parser.add_argument("--dim", type=int, default=CascadeConfig.dim)
    parser.add_argument("--depth", type=int, default=CascadeConfig.depth)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = ROOT/args.output
    tokens = ROOT/args.tokens if args.tokens else output/"text-tokens.npz"
    if not tokens.exists():
        words, _, _, _, shards = load_shape_corpus(ROOT/args.corpus)
        output.mkdir(parents=True, exist_ok=True)
        print(json.dumps(cache_text_tokens(words, tokens, args.device, shards)), flush=True)
    config = replace(CascadeTrainConfig(), steps=args.steps, batch_size=args.batch_size, validate_every=args.validate_every)
    train_cascade(ROOT/args.corpus, tokens, output, config, replace(CascadeConfig(), dim=args.dim, depth=args.depth),
                  args.device, args.resume)


if __name__ == "__main__":
    main()
