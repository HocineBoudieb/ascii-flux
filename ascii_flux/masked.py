"""Experimental denoising directly on glyph/color tokens, with no image codec.

Masked-token learning and iterative confidence sampling are inspired by MaskGIT
(Chang et al., CVPR 2022). This convolutional network, temporal context and native
glyph vocabulary are our own baseline, not the published MaskGIT architecture.
"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MaskedConfig:
    width: int = 64
    text_dim: int = 768
    condition_dim: int = 256
    text_dropout: float = .15
    emergence_steps: int = 8
    motion_steps: int = 4
    guidance: float = 2.
    temperature: float = .7

    def __post_init__(self):
        if self.width % 8 or min(self.width, self.text_dim, self.condition_dim, self.emergence_steps, self.motion_steps) < 1:
            raise ValueError("Invalid masked model dimensions")
        if not 0 <= self.text_dropout < 1 or self.temperature <= 0 or self.guidance < 0:
            raise ValueError("Invalid masked model sampling settings")


class ConditionedBlock(nn.Module):
    def __init__(self, channels, condition):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.film = nn.Linear(condition, channels*2)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value, condition):
        hidden = self.conv1(F.silu(self.norm1(value)))
        scale, shift = self.film(condition).chunk(2, -1)
        hidden = self.norm2(hidden)*(1+scale[:, :, None, None]) + shift[:, :, None, None]
        return (value+self.conv2(F.silu(hidden)))/math.sqrt(2)


class MaskedGlyphFlow(nn.Module):
    """State is bounded native tokens, masks, previous tokens and sampling phase.

    Channels: current glyph/color, unresolved mask, previous glyph/color,
    phase, mode. Internal mask IDs never appear in the rendered output.
    """
    def __init__(self, contract, config=None):
        super().__init__()
        self.contract, self.config = contract, config or MaskedConfig()
        w, d = self.config.width, self.config.condition_dim
        self.mask_glyph, self.mask_color = len(contract.glyphs), len(contract.palette)+1
        self.glyph_embedding = nn.Embedding(self.mask_glyph+1, 24)
        self.color_embedding = nn.Embedding(self.mask_color+1, 24)
        self.text_condition = nn.Sequential(nn.Linear(self.config.text_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.ratio_condition = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.mode_embedding = nn.Embedding(3, d)
        self.input = nn.Conv2d(99, w, 3, padding=1)
        self.block1 = ConditionedBlock(w, d)
        self.down1 = nn.Conv2d(w, w*2, 3, stride=2, padding=1)
        self.block2 = ConditionedBlock(w*2, d)
        self.down2 = nn.Conv2d(w*2, w*3, 3, stride=2, padding=1)
        self.block3 = ConditionedBlock(w*3, d)
        self.global_norm = nn.LayerNorm(w*3)
        self.global_attention = nn.MultiheadAttention(w*3, 4, batch_first=True)
        self.block4 = ConditionedBlock(w*3, d)
        self.up1 = nn.Conv2d(w*5, w*2, 3, padding=1)
        self.block5 = ConditionedBlock(w*2, d)
        self.up2 = nn.Conv2d(w*3, w, 3, padding=1)
        self.block6 = ConditionedBlock(w, d)
        self.glyph_head = nn.Conv2d(w, len(contract.glyphs), 1)
        self.color_head = nn.Conv2d(w, len(contract.palette), 1)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, contract.rows), torch.linspace(-1, 1, contract.cols), indexing="ij")
        self.register_buffer("coordinates", torch.stack((xx, yy))[None])

    def _color_ids(self, glyphs, colors):
        return torch.where(glyphs == self.mask_glyph, self.mask_color,
                           torch.where(glyphs == 0, len(self.contract.palette), colors)).long()

    def forward(self, glyphs, colors, previous_glyphs, previous_colors, text, ratio, mode):
        glyphs, previous_glyphs = glyphs.long(), previous_glyphs.long()
        colors = self._color_ids(glyphs, colors)
        previous_colors = self._color_ids(previous_glyphs, previous_colors)
        embed = torch.cat([self.glyph_embedding(glyphs), self.color_embedding(colors),
                           self.glyph_embedding(previous_glyphs), self.color_embedding(previous_colors)], -1).permute(0, 3, 1, 2)
        known = (glyphs != self.mask_glyph).float()[:, None]
        value = self.input(torch.cat([embed, known, self.coordinates.expand(glyphs.shape[0], -1, -1, -1)], 1))
        condition = self.text_condition(text*math.sqrt(self.config.text_dim)) + self.ratio_condition(ratio[:, None]) + self.mode_embedding(mode.long())
        a = self.block1(value, condition)
        b = self.block2(self.down1(a), condition)
        c = self.block3(self.down2(b), condition)
        tokens = c.flatten(2).transpose(1, 2)
        normalized = self.global_norm(tokens)
        attended, _ = self.global_attention(normalized, normalized, normalized, need_weights=False)
        c = self.block4((tokens+attended).transpose(1, 2).reshape_as(c), condition)
        b = self.block5(self.up1(torch.cat([F.interpolate(c, size=b.shape[-2:], mode="nearest"), b], 1)), condition)
        a = self.block6(self.up2(torch.cat([F.interpolate(b, size=a.shape[-2:], mode="nearest"), a], 1)), condition)
        return self.glyph_head(a), self.color_head(a)

    def initial_state(self, batch=1, noisy=False):
        state = self.coordinates.new_zeros((batch, 7, self.contract.rows, self.contract.cols))
        state[:, 1] = self.contract.empty_color
        state[:, 2] = 1
        state[:, 3] = self.mask_glyph
        state[:, 4] = self.mask_color
        return state

    def recondition(self, state):
        # Preserve the displayed grid; only restart the refinement schedule.
        result = state.clone()
        result[:, 3:5] = state[:, :2]
        result[:, 2] = (torch.rand_like(state[:, 2]) < .85).float()
        result[:, 5] = 0
        result[:, 6] = 2
        return result

    def tokens(self, state):
        glyphs = state[:, 0].long()
        colors = state[:, 1].long()
        return glyphs, torch.where(glyphs != 0, colors, self.contract.empty_color)

    def step(self, state, text):
        state = state.clone()
        phase = int(state[0, 5, 0, 0].item())
        mode = int(state[0, 6, 0, 0].item())
        total = self.config.motion_steps if mode == 1 else self.config.emergence_steps
        if phase >= total:
            state[:, 3:5] = state[:, :2]
            state[:, 2] = (torch.rand_like(state[:, 2]) < .65).float()
            state[:, 5] = 0
            state[:, 6] = 1
            phase, mode, total = 0, 1, self.config.motion_steps
        masked = state[:, 2].bool()
        g = state[:, 0].long().masked_fill(masked, self.mask_glyph)
        c = state[:, 1].long().masked_fill(masked, self.mask_color)
        pg, pc = state[:, 3].long(), state[:, 4].long()
        ratio = masked.float().mean((1, 2))
        modes = torch.full((len(state),), mode, device=state.device, dtype=torch.long)
        if self.config.guidance != 1:
            gl, cl = self(torch.cat([g,g]), torch.cat([c,c]), torch.cat([pg,pg]), torch.cat([pc,pc]),
                          torch.cat([text,torch.zeros_like(text)]), torch.cat([ratio,ratio]), torch.cat([modes,modes]))
            gl, ungl = gl.chunk(2); cl, uncl = cl.chunk(2)
            gl = ungl+self.config.guidance*(gl-ungl)
            cl = uncl+self.config.guidance*(cl-uncl)
        else:
            gl, cl = self(g,c,pg,pc,text,ratio,modes)
        temperature = self.config.temperature*(.25 if mode == 1 else max(.25, 1-phase/total))
        gp = F.softmax(gl.float(), 1)
        # Gumbel-max samples native tokens, rather than independently taking the
        # marginal argmax of every cell for every pose of a concept.
        randg = -torch.log(-torch.log(torch.rand_like(gl).clamp_(1e-6, 1-1e-6)))
        randc = -torch.log(-torch.log(torch.rand_like(cl).clamp_(1e-6, 1-1e-6)))
        sampled_g = (gl.float()+temperature*randg).argmax(1)
        sampled_c = (cl.float()+temperature*randc).argmax(1)
        sampled_c = torch.where(sampled_g != 0, sampled_c, self.contract.empty_color)
        state[:, 0] = torch.where(masked, sampled_g, state[:, 0].long())
        state[:, 1] = torch.where(masked, sampled_c, state[:, 1].long())
        confidence = gp.gather(1, sampled_g[:, None]).squeeze(1)
        next_ratio = math.cos((phase+1)/total*math.pi/2)
        # Keep positive anchors as well as background anchors: a vocabulary
        # dominated by spaces otherwise accepts only background in early steps.
        remaining = torch.zeros_like(masked)
        for i in range(len(state)):
            for foreground in (False, True):
                group = masked[i] & ((sampled_g[i] != 0) == foreground)
                indices = group.flatten().nonzero().flatten()
                count = min(len(indices), round(len(indices)*next_ratio)) if phase+1 < total else 0
                if count:
                    scores = confidence[i].flatten()[indices]
                    keep_masked = indices[scores.topk(count, largest=False).indices]
                    remaining[i].view(-1)[keep_masked] = True
        state[:, 2] = remaining.float()
        state[:, 5] = phase+1
        return state, gl, cl

    def config_dict(self):
        return asdict(self.config)


def masked_loss(gl, cl, glyphs, colors, mask):
    glyph = F.cross_entropy(gl.float(), glyphs.long(), reduction="none")
    glyph_loss = glyph[mask].mean()
    active = mask & (glyphs != 0)
    color_loss = F.cross_entropy(cl.permute(0,2,3,1)[active].float(), colors[active].long()) if active.any() else cl.sum()*0
    return glyph_loss+color_loss
