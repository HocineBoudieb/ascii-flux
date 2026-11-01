"""Resumable offline teacher shape corpus (tagged synthetic: teacher-t2i)."""
import argparse
import os
from pathlib import Path
import sys
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ascii_flux.shape_corpus import build_shape_corpus

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/v5/shapes-001")
    parser.add_argument("--max-words", type=int)
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--shard-words", type=int, default=100)
    args = parser.parse_args()
    build_shape_corpus(ROOT/args.output, ROOT/"data/lvvis/downloads/train_instances.json", args.keep, args.attempts,
                       args.shard_words, args.max_words)


if __name__ == "__main__":
    main()
