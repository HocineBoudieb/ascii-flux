"""Exercise the real NCA forward/backward; this is a hardware test, not F1."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from ascii_flux.contract import DEFAULT
from ascii_flux.model import GlyphNCA, token_loss
from ascii_flux.storage import write_json


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("No usable GPU in this interpreter; training has not started")
    model = GlyphNCA(DEFAULT).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    result = {"torch": str(torch.__version__), "hip": torch.version.hip,
              "device": torch.cuda.get_device_name(0),
              "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
              "parameters": sum(p.numel() for p in model.parameters()),
              "scope": "analytical hardware forward/backward test; no semantic training"}
    for precision in ("float32", "float16"):
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        text = torch.zeros(1, 768, device="cuda")
        target = torch.ones((1, DEFAULT.rows, DEFAULT.cols), dtype=torch.long, device="cuda")
        before = model.update[-1].weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=precision == "float16"):
            state = model.initial_state()
            for _ in range(4):
                state, glyphs, colors = model.step(state, text)
            loss = token_loss(glyphs, colors, target, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        if torch.equal(before, model.update[-1].weight) or not torch.isfinite(loss):
            raise RuntimeError("GPU failed the optimizer update test")
        result[precision] = {"backward_and_update": True, "loss": float(loss.detach()),
                             "seconds_including_first_compilation": time.monotonic()-started,
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        print(json.dumps(result[precision]), flush=True)
    write_json(Path(__file__).resolve().parents[1]/"outputs/gpu-verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
