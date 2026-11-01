"""Resumeable, explicitly experimental native-token training on real data."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ascii_flux.masked import MaskedConfig
from ascii_flux.masked_training import MaskedTrainConfig,train_masked


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="data/runs/masked-six-001")
    parser.add_argument("--steps",type=int,default=6000)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--memorize-one-frame",action="store_true",
                        help="Capacity diagnostic only, not a generalization result")
    parser.add_argument("--open-vocabulary",action="store_true",
                        help="Train on every label of the corpus instead of the six default subjects")
    parser.add_argument("--phrases",help="Phrase embedding cache from scripts/embed_phrases.py")
    parser.add_argument("--width",type=int,default=MaskedConfig.width)
    parser.add_argument("--batch-size",type=int,default=MaskedTrainConfig.batch_size)
    parser.add_argument("--validate-every",type=int,default=250)
    args = parser.parse_args()
    data = ROOT/"data/v5/million-001"
    output = ROOT/args.output
    config = replace(MaskedTrainConfig(),steps=args.steps,validate_every=args.validate_every,batch_size=args.batch_size,
                     memorize_one_frame=args.memorize_one_frame,
                     **({"subjects":()} if args.open_vocabulary else {}))
    train_masked(data/"manifest-million-001.json",data/"text-million-001.npz",
                 data/"reviews-million-001.json",output,config,replace(MaskedConfig(),width=args.width),
                 resume=output/"latest.pt" if args.resume else None,
                 phrases_path=ROOT/args.phrases if args.phrases else None)


if __name__ == "__main__":main()
