"""Instrumental measurements; semantic recognition requires a human study."""
import time
import numpy as np
import torch
from .storage import write_json


@torch.inference_mode()
def measure_rollout(model, embedding, steps=600, warmup=10, switch_embedding=None, switch_at=None):
    if steps < 500:
        raise ValueError("Stability diagnostics require at least 500 steps")
    device = next(model.parameters()).device
    text = torch.as_tensor(embedding, dtype=torch.float32, device=device).reshape(1, -1)
    other = None if switch_embedding is None else torch.as_tensor(switch_embedding, dtype=torch.float32, device=device).reshape(1, -1)
    model.eval()
    state = model.initial_state()
    for _ in range(warmup):
        state, _, _ = model.step(state, text)
    state = model.initial_state()
    latencies, changes, occupancy, norms, grids, transitions = [], [], [], [], set(), []
    previous = None
    finite = True
    for i in range(steps):
        condition = other if other is not None and i >= (switch_at or steps//2) else text
        if other is not None and i == (switch_at or steps//2) and hasattr(model,"recondition"):
            state = model.recondition(state)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        begin = time.perf_counter()
        state, _, _ = model.step(state, condition)
        glyphs, _ = model.tokens(state)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        latencies.append((time.perf_counter()-begin)*1000)
        if not torch.isfinite(state).all():
            finite = False
            break
        g = glyphs.cpu().numpy()[0]
        grids.add(g.tobytes())
        occupancy.append(float((g != 0).mean()))
        norms.append(float(state.abs().max()))
        if previous is not None:
            change = int((g != previous).sum())
            changes.append(change)
            transitions.append(change > 0)
        previous = g
    result = {"scope": "instrumental rollout diagnostics; not human recognition", "steps_requested": steps,
              "steps_measured": len(latencies), "device": str(device), "torch_threads": torch.get_num_threads(),
              "step_and_readout_ms_median": float(np.median(latencies)), "step_and_readout_ms_p95": float(np.percentile(latencies, 95)),
              "measured_max_steps_per_second": 1000/float(np.mean(latencies)),
              "glyph_changes_per_step_mean": float(np.mean(changes)) if changes else 0.,
              "distinct_updates_per_step": float(np.mean(transitions)) if transitions else 0.,
              "unique_grid_count": len(grids), "occupancy_first": occupancy[0] if occupancy else None,
              "occupancy_last": occupancy[-1] if occupancy else None,
              "state_abs_max": max(norms) if norms else None, "finite": finite,
              "frozen_last_100_steps": not any(changes[-100:]) if len(changes) >= 100 else None,
              "human_recognition": "not measured", "time_to_readable_seconds": None,
              "semantic_transition_seconds": None, "transition_continuity": "not measured",
              "ten_minute_stability": "not established by this step-count diagnostic"}
    return result


def export_onnx(model, output):
    """One recurrent step, including both token heads; noise and mask are inputs."""
    if not hasattr(model.config,"update_probability"):
        raise ValueError("This experimental architecture has no validated recurrent ONNX export yet")
    import onnxruntime as ort
    from .storage import atomic_file
    model = model.cpu().eval()
    state = model.initial_state()
    text = torch.zeros(1, model.config.text_dim)
    noise = torch.randn(1, 1, model.contract.rows, model.contract.cols)
    mask = (torch.rand_like(noise) < model.config.update_probability).float()
    with torch.inference_mode():
        expected = model(state, text, noise, mask)
    # Fixed grid/batch initially. External inputs provide all stochasticity.
    import io
    buffer = io.BytesIO()
    torch.onnx.export(model, (state, text, noise, mask), buffer, input_names=["state", "text", "noise", "update_mask"],
                      output_names=["next_state", "glyph_logits", "color_logits"], opset_version=17, dynamo=False)
    session = ort.InferenceSession(buffer.getvalue(), providers=["CPUExecutionProvider"])
    actual = session.run(None, {k: v.numpy() for k, v in zip(["state", "text", "noise", "update_mask"], (state, text, noise, mask))})
    errors = [float(np.max(np.abs(a-e.numpy()))) for a, e in zip(actual, expected)]
    if any(not np.allclose(a, e.numpy(), rtol=1e-4, atol=1e-5) for a, e in zip(actual, expected)):
        raise ValueError(f"ONNX numerical parity failed: {errors}")
    with atomic_file(output) as stream:
        stream.write(buffer.getvalue())
    write_json(str(output)+".json", {"contract": model.contract.to_dict(), "model_config": model.config_dict(),
                                   "onnx_max_abs_errors": errors, "scope": "export parity, not browser performance"})
    return errors
