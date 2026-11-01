from dataclasses import replace
import importlib
import json
import random
import sys
import threading
import time
import urllib.request
import numpy as np
import pytest
import torch
from ascii_flux.contract import DEFAULT
from ascii_flux.model import GlyphNCA, ModelConfig, token_loss
from ascii_flux.runtime import FlowRuntime, make_server
from ascii_flux.training import Corpus, TrainConfig, load_inference, train
from tests.fixtures import make_corpus


def tiny_model():
    return GlyphNCA(replace(DEFAULT, version="runtime-test", cols=16, rows=8), ModelConfig(state_channels=8, width=16))


def test_model_has_two_heads_and_masks_empty_color_gradients():
    model = tiny_model()
    state, gl, cl = model.step(model.initial_state(), torch.zeros(1, 768))
    assert gl.shape == (1, 7, 8, 16) and cl.shape == (1, 64, 8, 16)
    cl.retain_grad()
    loss = token_loss(gl, cl, torch.zeros(1, 8, 16, dtype=torch.long), torch.full((1, 8, 16), 255))
    loss.backward()
    assert not torch.any(cl.grad)
    assert 1_000_000 <= sum(p.numel() for p in GlyphNCA(DEFAULT).parameters()) <= 10_000_000


def test_foreground_weight_reduces_background_dominance():
    # One subject cell among eight empty cells must receive greater influence
    # when the foreground weight is enabled, with no target on empty colors.
    target = torch.zeros((1, 3, 3), dtype=torch.long);target[0, 1, 1] = 1
    colors = torch.full_like(target, 255);colors[0, 1, 1] = 1
    glyph = torch.zeros((1, 7, 3, 3), requires_grad=True)
    color = torch.zeros((1, 64, 3, 3), requires_grad=True)
    a = torch.autograd.grad(token_loss(glyph, color, target, colors), glyph, retain_graph=True)[0]
    b = torch.autograd.grad(token_loss(glyph, color, target, colors, foreground_weight=4), glyph)[0]
    assert b[0, 1, 1, 1].abs() > a[0, 1, 1, 1].abs()
    assert b[0, 0, 0, 0].abs() < a[0, 0, 0, 0].abs()


def test_real_backprop_and_exact_resume(tmp_path):
    manifest, embeddings = make_corpus(tmp_path/"corpus")
    model_config = ModelConfig(state_channels=8, width=16)
    config = TrainConfig(steps=2, batch_size=1, emergence_steps=2, rollout_frames=2, transition_steps=2,
                         val_batches=1, validate_every=1, long_every=99)
    with pytest.raises(ValueError, match="Calibrate"):
        Corpus(manifest, embeddings)
    with pytest.raises(ValueError, match="Synthetic"):
        Corpus(manifest, embeddings, allow_unreviewed=True)
    train(manifest, embeddings, tmp_path/"full", config, model_config, allow_unreviewed=True, allow_fixture=True)
    train(manifest, embeddings, tmp_path/"resumed", replace(config, steps=1), model_config, allow_unreviewed=True, allow_fixture=True)
    train(manifest, embeddings, tmp_path/"resumed", config, model_config, resume=tmp_path/"resumed/latest.pt", allow_unreviewed=True, allow_fixture=True)
    a = torch.load(tmp_path/"full/latest.pt", weights_only=False)
    b = torch.load(tmp_path/"resumed/latest.pt", weights_only=False)
    assert a["step"] == b["step"] == 2 and a["history"] == b["history"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    assert not a["provenance"]["recognition_validated"]
    assert a["best_model"] is not None and a["optimizer"] and a["rng"]
    with pytest.raises(ValueError, match="Development"):
        load_inference(tmp_path/"full/best.pt")
    model, provenance = load_inference(tmp_path/"full/best.pt", allow_development=True)
    assert provenance["synthetic_encoder"]
    assert torch.load(tmp_path/"full/best.pt", weights_only=True)["step"] == a["best_step"]


class CountingEncoder:
    def __init__(self, delay=0):
        self.calls=[]
        self.delay=delay

    def encode(self, prompts):
        self.calls.extend(prompts)
        time.sleep(self.delay)
        return np.ones((1, 768), np.float32)*.01


def test_condition_revision_refines_existing_grid_once_and_explicit_reset_takes_priority():
    model = tiny_model()
    observations = []
    def recondition(state):
        observations.append(state.clone())
        return state
    model.recondition = recondition
    runtime = FlowRuntime(model,CountingEncoder(),debounce=0.)
    sentinel = runtime.state.clone().fill_(.123)
    runtime.state = sentinel.clone()
    worker = threading.Thread(target=runtime._encode_loop,daemon=True)
    runtime.threads = [worker];worker.start()
    try:
        runtime.set_prompt('new concept')
        wait_for(lambda:runtime.applied_revision == 1)
        assert not observations
        runtime.tick();runtime.tick()
        assert len(observations) == 1
        torch.testing.assert_close(observations[0],sentinel,rtol=0,atol=0)
        runtime.condition_pending = True
        runtime.reset();runtime.tick()
        assert len(observations) == 1 and not torch.any(runtime.state)
    finally:
        runtime.close()


def wait_for(predicate, timeout=4):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("Asynchronous condition timed out")


def test_debounce_keeps_state_and_encoder_does_not_block_steps():
    model = tiny_model()
    encoder = CountingEncoder(.12)
    runtime = FlowRuntime(model, encoder, fps=30, debounce=.04)
    # Run text worker separately to prove conditioning changes cannot touch state.
    sentinel = runtime.state.clone().fill_(.123)
    runtime.state = sentinel.clone()
    worker=threading.Thread(target=runtime._encode_loop, daemon=True)
    runtime.threads=[worker];worker.start()
    try:
        runtime.set_prompt("CAFE\u0301")
        runtime.set_prompt("Chat")
        wait_for(lambda: runtime.applied_revision == 2)
        assert encoder.calls == ["chat"]
        torch.testing.assert_close(runtime.state, sentinel, rtol=0, atol=0)
        runtime.set_prompt("chat")
        time.sleep(.07)
        assert encoder.calls == ["chat"]
        runtime.tick()
        runtime.reset();runtime.tick()
        # Initialized update rule is zero: a reset explicitly clears hidden state.
        assert not torch.any(runtime.state)
    finally:
        runtime.close()
    runtime=FlowRuntime(tiny_model(), CountingEncoder(.25), fps=30, debounce=.01)
    runtime.start()
    try:
        runtime.set_prompt("slow encoder")
        wait_for(lambda: len(runtime.encoder.calls) == 1)
        initial=runtime.steps
        time.sleep(.15)
        assert runtime.steps > initial and runtime.applied_revision == 0
        wait_for(lambda: runtime.applied_revision == 1)
    finally:
        runtime.close()


def test_local_http_api_and_direct_token_player():
    runtime=FlowRuntime(tiny_model(),CountingEncoder())
    server=make_server(runtime, {"allow_unreviewed": True}, port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f"http://127.0.0.1:{server.server_port}"
    try:
        page=urllib.request.urlopen(url).read().decode()
        assert "fillText" in page
        req=urllib.request.Request(url+"/prompt", data=json.dumps({"prompt":"cat"}).encode(), headers={"Content-Type":"application/json"})
        assert json.load(urllib.request.urlopen(req))["revision"] == 1
        frame=json.load(urllib.request.urlopen(url+"/frame"))
        assert len(frame["glyphs"])==8 and frame["steps"]==0
    finally:
        server.shutdown();server.server_close();thread.join(2)


def test_every_keystroke_can_apply_without_reset_and_late_inputs_are_ignored():
    model = tiny_model()
    encoder = CountingEncoder()
    runtime = FlowRuntime(model, encoder, debounce=0)
    sentinel = runtime.state.clone().fill_(.37)
    runtime.state = sentinel.clone()
    worker = threading.Thread(target=runtime._encode_loop, daemon=True)
    runtime.threads = [worker];worker.start()
    try:
        for sequence, prompt in enumerate(("c", "ca", "cat"), 1):
            assert runtime.set_prompt(prompt, "typing-test", sequence) == sequence
            wait_for(lambda: runtime.applied_revision == sequence)
            torch.testing.assert_close(runtime.state, sentinel, rtol=0, atol=0)
        assert encoder.calls == ["c", "ca", "cat"]
        assert runtime.set_prompt("c", "typing-test", 1) == 3
        assert runtime.prompt == "cat" and runtime.snapshot()["input_sequence"] == 3
        with pytest.raises(ValueError, match="sequence"):
            runtime.set_prompt("x", "typing-test", True)
    finally:
        runtime.close()


def test_slow_stale_embedding_cannot_replace_latest_typed_text():
    class SlowEncoder:
        def encode(self, prompts):
            time.sleep(.07)
            return np.full((1, 768), len(prompts[0]), np.float32)
    runtime = FlowRuntime(tiny_model(), SlowEncoder(), debounce=0)
    runtime.start()
    try:
        runtime.set_prompt("c", "typing-test", 1)
        time.sleep(.025)
        steps = runtime.steps
        runtime.set_prompt("cat", "typing-test", 2)
        wait_for(lambda: runtime.applied_revision == 2)
        assert torch.all(runtime.text == 3)
        assert runtime.steps > steps
    finally:
        runtime.close()


def test_onnx_export_numerical_parity(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from ascii_flux.evaluation import export_onnx
    model=tiny_model()
    torch.nn.init.normal_(model.update[-1].weight, std=.01)
    errors=export_onnx(model,tmp_path/"step.onnx")
    assert max(errors)<1e-4


def test_long_rollout_reports_frozen_state_as_failure():
    from ascii_flux.evaluation import measure_rollout
    result=measure_rollout(tiny_model(), np.zeros(768,np.float32), steps=500, warmup=1)
    assert result["finite"] and result["frozen_last_100_steps"]
    assert result["unique_grid_count"]==1
    assert result["human_recognition"]=="not measured" and result["time_to_readable_seconds"] is None
