"""Train/resume the real-data native causal glyph experiment."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ascii_flux.causal_training import CausalTrainConfig,train_causal


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="data/runs/causal-six-001")
    parser.add_argument("--steps",type=int,default=4000)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--memorize-one-frame",action="store_true")
    args = parser.parse_args()
    data,output = root/"data/v5/million-001",root/args.output
    config = replace(CausalTrainConfig(),steps=args.steps,memorize_one_frame=args.memorize_one_frame)
    train_causal(data/"manifest-million-001.json",data/"text-million-001.npz",data/"reviews-million-001.json",
                 output,config,resume=output/"latest.pt" if args.resume else None)


if __name__ == "__main__":main()
