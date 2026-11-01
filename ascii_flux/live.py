"""Live full-screen demo of the cascade model: one sampling step per frame.

The grid is the model state. A prompt revision is encoded once (separate
thread, short debounce) and re-conditions the current grid without clearing
it: part of the silhouette is re-masked and the schedule resumes, so the old
shape turns into the new one. Once a shape is complete the grid holds still;
no motion has been learned yet. Browser receives tokens over Server-Sent Events.
"""
import base64
import json
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
from .contract import normalize_text


class LiveCascade:
    def __init__(self, model, encoder, fps=24., debounce=.12):
        self.model, self.encoder, self.fps, self.debounce = model.eval(), encoder, fps, debounce
        self.device = next(model.parameters()).device
        c, n = model.config, model.ph*model.pw
        self.s_total, self.g_total = c.silhouette_steps, c.glyph_steps
        self.colon = model.contract.glyphs.index(":")
        self.dim_color = len(model.contract.palette)-1
        self.lock = threading.Condition()
        self.stopped = threading.Event()
        self.prompt, self.revision, self.revised_at = "", 0, 0.
        self.pending_text, self.text = None, None
        self.version, self.frame = 0, None
        self.silhouette = torch.zeros((1, n), dtype=torch.long, device=self.device)
        self.display_g = torch.zeros((1, model.contract.rows, model.contract.cols), dtype=torch.long, device=self.device)
        self.display_c = torch.full_like(self.display_g, model.contract.empty_color)
        self.phase, self.step = "idle", 0
        self._publish()

    # ----- prompt handling -----
    def set_prompt(self, prompt):
        prompt = normalize_text(prompt)[:200]
        with self.lock:
            if prompt != self.prompt:
                self.prompt, self.revised_at = prompt, time.monotonic()
                self.revision += 1
                self.lock.notify_all()

    def _encode_loop(self):
        done = 0
        while not self.stopped.is_set():
            with self.lock:
                wait = self.debounce-(time.monotonic()-self.revised_at)
                if done == self.revision or wait > 0:
                    self.lock.wait(.05 if done == self.revision else wait)
                    continue
                revision, prompt = self.revision, self.prompt
            done = revision
            text = None if not prompt else torch.tensor(
                self.encoder.encode_tokens([prompt], self.model.config.text_tokens), device=self.device).float()
            with self.lock:
                if revision == self.revision:
                    self.pending_text = ("clear",) if text is None else ("text", text)

    # ----- model clock -----
    def _recondition(self, text):
        """New concept: the model redraws its silhouette from scratch while the
        displayed grid is kept; cells change only as the model decides them."""
        self.text = text
        n = self.silhouette.shape[1]
        self.silhouette = torch.full((1, n), 2, dtype=torch.long, device=self.device)
        self.s_masked = torch.ones((1, n), dtype=torch.bool, device=self.device)
        self.step = 0
        self.phase = "silhouette"

    @torch.inference_mode()
    def tick(self):
        with self.lock:
            pending, self.pending_text = self.pending_text, None
        if pending is not None:
            if pending[0] == "clear":
                self.phase, self.text = "idle", None
                self.silhouette.zero_()
                self.display_g.zero_()
                self.display_c.fill_(self.model.contract.empty_color)
                self._publish()
                return
            self._recondition(pending[1])
        m = self.model
        if self.phase == "silhouette":
            self.silhouette, self.s_masked = m.silhouette_step(self.text, self.silhouette, self.s_masked, self.step, self.s_total)
            self.step += 1
            cells = lambda value: m.grid((self.silhouette == value)[..., None].expand(-1, -1, m.cells))
            occupied, vacated = cells(1), cells(0)
            # Undecided cells keep what is shown; cells decided empty vanish;
            # cells decided occupied keep their glyph or appear as a faint fill.
            self.display_g = torch.where(vacated, 0, torch.where(occupied & (self.display_g == 0), self.colon, self.display_g))
            self.display_c = torch.where(self.display_g == 0, m.contract.empty_color,
                                         torch.where(self.display_c == m.contract.empty_color, self.dim_color, self.display_c))
            if self.step >= self.s_total:
                self.glyphs, self.colors, self.g_masked = m.start_glyphs(self.silhouette)
                self.phase, self.step = "glyphs", 0
        elif self.phase == "glyphs":
            self.glyphs, self.colors, self.g_masked = m.glyph_step(self.text, self.glyphs, self.colors, self.silhouette,
                                                                   self.g_masked, self.step, self.g_total)
            self.step += 1
            g, c = m._render(self.silhouette, self.glyphs, self.colors)
            decided = m.grid((~self.g_masked)[..., None].expand(-1, -1, m.cells)) & (m.grid((self.silhouette == 1)[..., None].expand(-1, -1, m.cells)))
            self.display_g = torch.where(decided, g, self.display_g)
            self.display_c = torch.where(decided, c, self.display_c)
            if self.step >= self.g_total:
                self.display_g, self.display_c = g, c
                self.phase = "idle"
        else:
            return
        self._publish()

    def _publish(self):
        glyphs = "".join(self.model.contract.glyphs[i] for i in self.display_g[0].flatten().tolist())
        colors = base64.b64encode(bytes(self.display_c[0].flatten().to(torch.uint8).tolist())).decode()
        with self.lock:
            self.version += 1
            self.frame = json.dumps({"v": self.version, "g": glyphs, "c": colors})
            self.lock.notify_all()

    def _clock(self):
        deadline = time.monotonic()
        while not self.stopped.is_set():
            self.tick()
            deadline = max(deadline+1/self.fps, time.monotonic())
            self.stopped.wait(max(0., deadline-time.monotonic()))

    def start(self):
        for target in (self._encode_loop, self._clock):
            threading.Thread(target=target, daemon=True).start()

    def wait_frame(self, after, timeout=1.):
        with self.lock:
            if self.version == after:
                self.lock.wait(timeout)
            return self.version, self.frame


def make_server(live, port):
    page = Path(__file__).with_name("live.html").read_bytes()
    contract = live.model.contract
    config = json.dumps({"rows": contract.rows, "cols": contract.cols, "cell": contract.cell_height_width,
                         "palette": [list(c) for c in contract.palette], "empty": contract.empty_color}).encode()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def _send(self, status, body, kind):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == "/config":
                self._send(200, config, "application/json")
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                seen = -1
                try:
                    while not live.stopped.is_set():
                        version, frame = live.wait_frame(seen)
                        if version != seen:
                            seen = version
                            self.wfile.write(f"data: {frame}\n\n".encode())
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
            else:
                self._send(404, b"{}", "application/json")

        def do_POST(self):
            origin = self.headers.get("Origin")
            if self.path != "/prompt" or (origin and origin != f"http://{self.headers.get('Host')}"):
                self._send(403, b"{}", "application/json")
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                value = json.loads(self.rfile.read(min(length, 4000)))
                live.set_prompt(str(value.get("prompt", "")))
                self._send(200, b"{}", "application/json")
            except ValueError:
                self._send(400, b"{}", "application/json")

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(checkpoint, port=8767, device="cuda", fps=24., on_ready=None):
    from .cascade import CascadeConfig, CascadeGlyphModel
    from .contract import Contract
    from .text import TextEncoder
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = CascadeGlyphModel(Contract.from_dict(saved["contract"]), CascadeConfig(**saved["model_config"]))
    model.load_state_dict(saved["model"])
    encoder = TextEncoder(device)
    model = model.to(device).eval()
    # Warm-up: compile GPU kernels before the first prompt, and time a step.
    begin = time.perf_counter()
    with torch.inference_mode():
        model.sample(torch.tensor(encoder.encode_tokens(["warm up"], model.config.text_tokens), device=device).float())
    print(f"warm-up {time.perf_counter()-begin:.1f}s", flush=True)
    live = LiveCascade(model, encoder, fps)
    server = make_server(live, port)
    live.start()
    print(f"ASCII Flux live: http://127.0.0.1:{port}", flush=True)
    if on_ready:
        on_ready(f"http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        live.stopped.set()
        server.server_close()
