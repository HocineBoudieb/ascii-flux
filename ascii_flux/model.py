"""Glyph-space NCA baseline. No image representation exists in this module."""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    state_channels: int = 48
    width: int = 256
    text_dim: int = 768
    update_probability: float = 0.5
    global_stride: int = 4
    text_dropout: float = 0.1

    def __post_init__(self):
        if min(self.state_channels, self.width, self.text_dim, self.global_stride) <= 0:
            raise ValueError("Invalid model dimensions")
        if not 0 < self.update_probability <= 1 or not 0 <= self.text_dropout < 1:
            raise ValueError("Invalid stochastic probabilities")


class GlyphNCA(nn.Module):
    def __init__(self, contract, config=None):
        super().__init__()
        self.contract = contract
        self.config = config or ModelConfig()
        c, w = self.config.state_channels, self.config.width
        self.glyph_embedding = nn.Embedding(len(contract.glyphs), c)
        self.color_embedding = nn.Embedding(len(contract.palette)+1, c, padding_idx=len(contract.palette))
        self.local = nn.Conv2d(c, c*2, 3, padding=1)
        self.global_context = nn.Sequential(nn.Conv2d(c, c*2, 3, padding=1), nn.SiLU(), nn.Conv2d(c*2, c*2, 3, padding=1))
        self.project = nn.Conv2d(c*4+3, w, 1)
        self.film = nn.Sequential(nn.Linear(self.config.text_dim, 768), nn.SiLU(), nn.Linear(768, w*2))
        self.update = nn.Sequential(nn.SiLU(), nn.Conv2d(w, w, 1), nn.SiLU(), nn.Conv2d(w, c, 1))
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)
        self.glyph_head = nn.Conv2d(c, len(contract.glyphs), 1)
        self.color_head = nn.Conv2d(c, len(contract.palette), 1)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, contract.rows), torch.linspace(-1, 1, contract.cols), indexing="ij")
        self.register_buffer("coordinates", torch.stack((xx, yy))[None])

    def initial_state(self, batch=1, noisy=False):
        state = self.coordinates.new_zeros((batch, self.config.state_channels, self.contract.rows, self.contract.cols))
        return state + torch.randn_like(state)*.01 if noisy else state

    def lift(self, glyphs, colors):
        # Teacher forcing only. The deployed recurrent step does not read targets.
        color = torch.where(glyphs != 0, colors, len(self.contract.palette))
        return (self.glyph_embedding(glyphs) + self.color_embedding(color)).permute(0, 3, 1, 2).contiguous()

    def forward(self, state, text, noise, update_mask):
        """Explicit noise/mask inputs permit deterministic ONNX parity and RNG control."""
        cfg = self.config
        local = self.local(state)
        pooled = F.avg_pool2d(state, cfg.global_stride)
        context = F.interpolate(self.global_context(pooled), size=(self.contract.rows, self.contract.cols), mode="nearest")
        coordinates = self.coordinates.expand(state.shape[0], -1, -1, -1)
        features = self.project(torch.cat((local, context, coordinates, noise), dim=1))
        scale, bias = self.film(text).chunk(2, dim=1)
        features = features * (1 + scale[:, :, None, None]) + bias[:, :, None, None]
        next_state = state + .1*torch.tanh(self.update(features))*update_mask
        return next_state, self.glyph_head(next_state), self.color_head(next_state)

    def step(self, state, text):
        shape = (state.shape[0], 1, self.contract.rows, self.contract.cols)
        noise = torch.randn(shape, device=state.device, dtype=state.dtype)
        mask = (torch.rand(shape, device=state.device) < self.config.update_probability).to(state.dtype)
        return self(state, text, noise, mask)

    def tokens(self, state):
        glyphs = self.glyph_head(state).argmax(1)
        colors = self.color_head(state).argmax(1)
        return glyphs, torch.where(glyphs != 0, colors, self.contract.empty_color)

    def config_dict(self):
        return asdict(self.config)


def token_loss(glyph_logits, color_logits, glyphs, colors, foreground_weight=1.):
    # Color on empty cells has neither a target nor a gradient. Separate heads.
    if foreground_weight == 1.:
        glyph_loss = F.cross_entropy(glyph_logits, glyphs)
    else:
        weights = torch.where(glyphs != 0, foreground_weight, 1.)
        glyph_loss = (F.cross_entropy(glyph_logits, glyphs, reduction="none")*weights).sum()/weights.sum()
    valid = glyphs != 0
    if valid.any():
        color_loss = F.cross_entropy(color_logits.permute(0, 2, 3, 1)[valid], colors[valid])
    else:
        color_loss = color_logits.sum()*0
    return glyph_loss + color_loss

