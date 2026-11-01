"""Full-screen live demo: type a word, the glyph grid evolves step by step."""
import argparse
import os
from pathlib import Path
import sys
import webbrowser
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ascii_flux.live import serve

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="data/runs/cascade-001/best.pt")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=24.)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    serve(ROOT/args.checkpoint, args.port, args.device, args.fps,
          on_ready=None if args.no_browser else webbrowser.open)


if __name__ == "__main__":
    main()
