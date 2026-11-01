"""Native recurrent inference, fixed step clock and independent text worker."""
import json
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
import numpy as np
import torch
from .contract import normalize_text


class FlowRuntime:
    def __init__(self, model, encoder, fps=24., debounce=.25):
        if fps <= 0 or debounce < 0:
            raise ValueError("Invalid runtime clock")
        self.model, self.encoder, self.fps, self.debounce = model.eval(), encoder, fps, debounce
        self.device = next(model.parameters()).device
        self.state = model.initial_state()
        self.text = torch.zeros(1, model.config.text_dim, device=self.device)
        self.lock = threading.Condition()
        self.stopped = threading.Event()
        self.revision, self.applied_revision = 0, 0
        self.prompt, self.revised_at, self.reset_pending = "", 0., False
        self.steps, self.dropped_ticks, self.step_ms, self.encode_ms = 0, 0, None, None
        self.error = None
        self.client_sequences = OrderedDict()
        self.input_client, self.input_sequence = None, None
        self.condition_pending = False
        self.frame = {"glyphs": np.zeros((model.contract.rows, model.contract.cols), np.uint8).tolist(),
                      "colors": np.full((model.contract.rows, model.contract.cols), model.contract.empty_color, np.uint8).tolist()}
        self.threads = []

    def set_prompt(self, prompt, client=None, sequence=None):
        prompt = normalize_text(prompt)
        if len(prompt) > 2000:
            raise ValueError("Prompt too long")
        if client is not None or sequence is not None:
            if not isinstance(client, str) or not 1 <= len(client) <= 80:
                raise ValueError("Invalid input session")
            if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
                raise ValueError("Invalid input sequence")
        with self.lock:
            if client is not None:
                if sequence <= self.client_sequences.get(client, -1):
                    return self.revision
                self.client_sequences[client] = sequence
                self.client_sequences.move_to_end(client)
                if len(self.client_sequences) > 32:
                    self.client_sequences.popitem(last=False)
                self.input_client, self.input_sequence = client, sequence
            if prompt != self.prompt:
                self.prompt, self.revised_at = prompt, time.monotonic()
                self.revision += 1
                self.error = None
                self.lock.notify_all()
            return self.revision

    def reset(self):
        with self.lock:
            self.reset_pending = True

    def _encode_loop(self):
        attempted = 0
        while not self.stopped.is_set():
            with self.lock:
                if attempted == self.revision:
                    self.lock.wait(.1)
                    continue
                delay = self.debounce - (time.monotonic()-self.revised_at)
                if delay > 0:
                    self.lock.wait(min(delay, .1))
                    continue
                revision, prompt = self.revision, self.prompt
                attempted = revision
            begin = time.perf_counter()
            try:
                vector = self.encoder.encode([prompt])
                if vector.shape != (1, self.model.config.text_dim) or not np.isfinite(vector).all():
                    raise ValueError("Invalid text embedding")
                vector = torch.as_tensor(vector, dtype=torch.float32, device=self.device)
                with self.lock:
                    if revision == self.revision:
                        self.text = vector
                        self.condition_pending = True
                        self.applied_revision = revision
                        self.encode_ms = (time.perf_counter()-begin)*1000
            except Exception as exc:
                with self.lock:
                    if revision == self.revision:
                        self.error = f"Text encoding failed: {exc}"

    @torch.inference_mode()
    def tick(self):
        with self.lock:
            text = self.text
            if self.reset_pending:
                self.state = self.model.initial_state()
                self.reset_pending = False
            elif self.condition_pending and hasattr(self.model,"recondition"):
                self.state = self.model.recondition(self.state)
            self.condition_pending = False
        begin = time.perf_counter()
        state, _, _ = self.model.step(self.state, text)
        if not torch.isfinite(state).all():
            raise FloatingPointError("Non-finite recurrent state")
        glyphs, colors = self.model.tokens(state)
        self.state = state
        frame = {"glyphs": glyphs[0].cpu().tolist(), "colors": colors[0].cpu().tolist()}
        with self.lock:
            self.frame = frame
            self.steps += 1
            self.step_ms = (time.perf_counter()-begin)*1000

    def _step_loop(self):
        deadline = time.monotonic()
        try:
            while not self.stopped.is_set():
                self.tick()
                deadline += 1/self.fps
                now = time.monotonic()
                if now > deadline:
                    missed = int((now-deadline)*self.fps)+1
                    with self.lock:
                        self.dropped_ticks += missed
                    deadline += missed/self.fps
                self.stopped.wait(max(0., deadline-time.monotonic()))
        except Exception as exc:
            with self.lock:
                self.error = f"Model stopped: {exc}"
            self.stopped.set()

    def snapshot(self):
        with self.lock:
            return {**self.frame, "steps": self.steps, "requested_fps": self.fps,
                    "dropped_ticks": self.dropped_ticks, "step_ms": self.step_ms, "encode_ms": self.encode_ms,
                    "revision": self.revision, "applied_revision": self.applied_revision, "error": self.error,
                    "input_client": self.input_client, "input_sequence": self.input_sequence}

    def start(self):
        if self.threads:
            raise RuntimeError("Runtime already started")
        self.threads = [threading.Thread(target=self._encode_loop, daemon=True),
                        threading.Thread(target=self._step_loop, daemon=True)]
        for thread in self.threads:
            thread.start()

    def close(self):
        self.stopped.set()
        with self.lock:
            self.lock.notify_all()
        for thread in self.threads:
            thread.join(timeout=2)


def make_server(runtime, provenance, port=8766):
    page = Path(__file__).with_name("player.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, data, content_type="application/json"):
            body = data if isinstance(data, bytes) else json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # The tab closed or canceled an obsolete request.

        def do_GET(self):
            if self.path == "/":
                self.send(200, page, "text/html")
            elif self.path == "/frame":
                self.send(200, runtime.snapshot())
            elif self.path == "/config":
                self.send(200, {"contract": runtime.model.contract.to_dict(), "provenance": provenance,
                                "application": "ascii-flux-native", "input_debounce_ms": runtime.debounce*1000,
                                "execution": "local Python model; browser renders glyph tokens", "recognition_validated": False})
            else:
                self.send(404, {"error": "Not found"})

        def do_POST(self):
            try:
                origin = self.headers.get("Origin")
                if origin and origin != f"http://{self.headers.get('Host')}":
                    self.send(403, {"error": "Origin mismatch"})
                    return
                if not self.headers.get("Content-Type", "").startswith("application/json"):
                    raise ValueError("JSON required")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16000:
                    raise ValueError("Invalid request size")
                value = json.loads(self.rfile.read(length))
                if self.path == "/prompt":
                    if not isinstance(value.get("prompt"), str):
                        raise ValueError("Prompt must be text")
                    self.send(200, {"revision": runtime.set_prompt(value["prompt"], value.get("client"), value.get("sequence"))})
                elif self.path == "/reset":
                    runtime.reset()
                    self.send(200, {"reset": "scheduled"})
                else:
                    self.send(404, {"error": "Not found"})
            except (ValueError, KeyError) as exc:
                self.send(400, {"error": str(exc)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(checkpoint, port=8766, fps=24., device="cpu", allow_development=False, debounce=0.):
    from .text import TextEncoder
    from .inference import load_inference
    model, provenance = load_inference(checkpoint, device, allow_development)
    if provenance.get("synthetic_encoder"):
        raise ValueError("A fixture checkpoint has no SigLIP semantics and cannot drive the prompt demo")
    encoder = TextEncoder(device)
    runtime = FlowRuntime(model, encoder, fps, debounce=debounce)
    server = make_server(runtime, provenance, port)
    runtime.start()
    print(f"ASCII Flux research prototype: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runtime.close()

