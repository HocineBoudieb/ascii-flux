"""Embed several generic phrasings of every corpus label with the frozen text encoder."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ascii_flux.text import cache_phrase_embeddings


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",default="data/v5/million-001/manifest-million-001.json")
    parser.add_argument("--output",default="data/v5/million-001/phrases-million-001.npz")
    parser.add_argument("--device",default="cpu")
    args = parser.parse_args()
    print(json.dumps(cache_phrase_embeddings(ROOT/args.manifest,ROOT/args.output,args.device),indent=2))


if __name__ == "__main__":main()
