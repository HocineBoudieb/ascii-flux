"""Controlled real-token diffusion experiment initialized from the masked run."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ascii_flux.discrete_training import train_discrete
from ascii_flux.masked_training import MaskedTrainConfig


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--output',default='data/runs/discrete-six-001')
    args = parser.parse_args()
    data,output = root/'data/v5/million-001',root/args.output
    config = replace(MaskedTrainConfig(),steps=args.steps,validate_every=250)
    train_discrete(data/'manifest-million-001.json',data/'text-million-001.npz',data/'reviews-million-001.json',
                   output,config,resume=output/'latest.pt' if args.resume else None,
                   warm_start=root/'data/runs/masked-six-001/best.pt')


if __name__ == '__main__':main()
