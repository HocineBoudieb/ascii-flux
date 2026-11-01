"""Two-stage masked token transformer over native glyph grids.

Design follows published masked generative transformers (MaskGIT, Chang et al.
2022; Muse, Chang et al. 2023: cross-attention on text token sequences,
classifier-free guidance) and cascaded generation (Imagen, Saharia et al.
2022): a silhouette is decided first, glyphs and colors are filled inside it.
One backbone is shared by both stages. Grids are cut into 2x2-cell patches.
Nothing here touches pixels: inputs and outputs are glyph/color token IDs.
"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

# Horizontal mirror of the structure vocabulary " .:+*#@-_|/\()'".
MIRROR = {"/": "\\", "\\": "/", "(": ")", ")": "("}


@dataclass(frozen=True)
class CascadeConfig:
    dim: int = 256
    depth: int = 8
    heads: int = 8
    text_dim: int = 768
    text_tokens: int = 16
    patch: int = 2
    text_dropout: float = .1
    silhouette_steps: int = 12
    glyph_steps: int = 8
    guidance: float = 3.
    temperature: float = 1.

    def __post_init__(self):
        if self.dim % self.heads or self.dim % (2*self.patch**2) or min(self.depth, self.text_tokens) < 1:
            raise ValueError("Invalid cascade dimensions")


class Attention(nn.Module):
    """Fused projections and scaled_dot_product_attention, without the layout
    copies of nn.MultiheadAttention (measured as the bottleneck on ROCm)."""
    def __init__(self, dim, heads, cross=False):
        super().__init__()
        self.heads, self.cross = heads, cross
        if cross:
            self.q, self.kv = nn.Linear(dim, dim), nn.Linear(dim, 2*dim)
        else:
            self.qkv = nn.Linear(dim, 3*dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, context=None):
        b, n, d = x.shape
        h = self.heads
        if self.cross:
            q = self.q(x).view(b, n, h, d//h).transpose(1, 2)
            k, v = self.kv(context).view(context.shape[0], context.shape[1], 2, h, d//h).permute(2, 0, 3, 1, 4)
            if k.shape[0] != b:
                k, v = k.expand(b, -1, -1, -1), v.expand(b, -1, -1, -1)
        else:
            q, k, v = self.qkv(x).view(b, n, 3, h, d//h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = Attention(dim, heads)
        self.cross = Attention(dim, heads, cross=True)
        self.mlp = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))

    def forward(self, x, text):
        x = x+self.attention(self.norm1(x))
        x = x+self.cross(self.norm2(x), text)
        return x+self.mlp(self.norm3(x))


class CascadeGlyphModel(nn.Module):
    def __init__(self, contract, config=None):
        super().__init__()
        self.contract, self.config = contract, config or CascadeConfig()
        c, p, d = self.config, self.config.patch, self.config.dim
        if contract.rows % p or contract.cols % p:
            raise ValueError("Grid must divide into patches")
        self.ph, self.pw = contract.rows//p, contract.cols//p
        self.cells = p*p
        self.glyph_count, self.color_count = len(contract.glyphs), len(contract.palette)
        self.mask_glyph = self.glyph_count            # input-only IDs
        self.empty_color_id, self.mask_color = self.color_count, self.color_count+1
        e = d//(2*self.cells)
        self.glyph_embedding = nn.Embedding(self.glyph_count+1, e)
        self.color_embedding = nn.Embedding(self.color_count+2, e)
        self.silhouette_embedding = nn.Embedding(3, d)     # empty, occupied, masked
        self.stage_embedding = nn.Embedding(2, d)
        self.position = nn.Parameter(torch.randn(1, self.ph*self.pw, d)*.02)
        self.text_projection = nn.Sequential(nn.LayerNorm(c.text_dim), nn.Linear(c.text_dim, d))
        self.text_position = nn.Parameter(torch.randn(1, c.text_tokens, d)*.02)
        self.null_text = nn.Parameter(torch.randn(1, c.text_tokens, d)*.02)
        self.blocks = nn.ModuleList(Block(d, c.heads) for _ in range(c.depth))
        self.norm = nn.LayerNorm(d)
        self.silhouette_head = nn.Linear(d, 2)
        self.glyph_head = nn.Linear(d, self.cells*self.glyph_count)
        self.color_head = nn.Linear(d, self.cells*self.color_count)
        mirror = [contract.glyphs.index(MIRROR.get(g, g)) for g in contract.glyphs]
        self.register_buffer("mirror_glyph", torch.tensor(mirror), persistent=False)

    # Layout helpers: grid [B,R,C] <-> patches [B,N,cells].
    def patches(self, grid):
        b, p = grid.shape[0], self.config.patch
        return grid.view(b, self.ph, p, self.pw, p).permute(0, 1, 3, 2, 4).reshape(b, self.ph*self.pw, self.cells)

    def grid(self, patches):
        b, p = patches.shape[0], self.config.patch
        return patches.view(b, self.ph, self.pw, p, p).permute(0, 1, 3, 2, 4).reshape(b, self.contract.rows, self.contract.cols)

    def silhouette_of(self, glyphs):
        return (self.patches(glyphs) != 0).any(-1).long()

    def text_context(self, text, drop=None):
        if text is None:
            return self.null_text
        context = self.text_projection(text.float())+self.text_position
        if drop is not None:
            context = torch.where(drop[:, None, None], self.null_text.expand_as(context), context)
        return context

    def backbone(self, x, stage, text):
        x = x+self.position+self.stage_embedding.weight[stage]
        text = text.expand(x.shape[0], -1, -1)
        for block in self.blocks:
            x = block(x, text)
        return self.norm(x)

    def silhouette_logits(self, silhouette, text):
        """silhouette [B,N] in {0,1,2=masked}; returns [B,N,2]."""
        return self.silhouette_head(self.backbone(self.silhouette_embedding(silhouette), 1, text))

    def glyph_logits(self, glyph_patches, color_patches, silhouette, text):
        """Patches [B,N,cells] with masked IDs; returns glyph [B,N,cells,G], color [B,N,cells,P]."""
        b, n = glyph_patches.shape[:2]
        colors = torch.where(glyph_patches == self.mask_glyph, self.mask_color,
                             torch.where(glyph_patches == 0, self.empty_color_id, color_patches))
        x = torch.cat([self.glyph_embedding(glyph_patches).flatten(2), self.color_embedding(colors).flatten(2)], -1)
        x = x+self.silhouette_embedding(silhouette)
        h = self.backbone(x, 0, text)
        return (self.glyph_head(h).view(b, n, self.cells, self.glyph_count),
                self.color_head(h).view(b, n, self.cells, self.color_count))

    def mirror(self, glyphs, colors):
        glyphs = self.mirror_glyph[glyphs.flip(-1).long()]
        return glyphs, colors.flip(-1)

    def config_dict(self):
        return asdict(self.config)

    # ----- sampling -----
    def _guided(self, function, text, *inputs):
        guidance = self.config.guidance
        if guidance == 1:
            return function(*inputs, self.text_context(text))
        doubled = [torch.cat([i, i]) for i in inputs]
        context = torch.cat([self.text_context(text), self.null_text.expand(len(text), -1, -1)])
        outputs = function(*doubled, context)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        guided = tuple(u+guidance*(c-u) for c, u in (o.float().chunk(2) for o in outputs))
        return guided if len(guided) > 1 else guided[0]

    @staticmethod
    def _keep_masked(confidence, masked, ratio, temperature):
        """MaskGIT schedule: re-mask the least confident of the newly sampled."""
        noise = -torch.log(-torch.log(torch.rand_like(confidence).clamp(1e-6, 1-1e-6)))
        score = confidence+temperature*noise
        score = torch.where(masked, score, torch.full_like(score, math.inf))
        count = (masked.sum(-1).float()*ratio).floor().long()
        order = score.argsort(-1)
        rank = torch.empty_like(order)
        rank.scatter_(-1, order, torch.arange(order.shape[-1], device=order.device).expand_as(order))
        return rank < count[:, None]

    @torch.no_grad()
    def silhouette_step(self, text, silhouette, masked, step, total):
        """One MaskGIT step of the silhouette stage; returns (silhouette, masked)."""
        logits = self._guided(self.silhouette_logits, text, silhouette)
        tau = self.config.temperature*(1-step/total)
        noise = -torch.log(-torch.log(torch.rand(logits.shape, device=logits.device).clamp(1e-6, 1-1e-6)))
        choice = (logits+tau*noise).argmax(-1)
        confidence = F.log_softmax(logits, -1).gather(-1, choice[..., None])[..., 0]
        candidate = torch.where(masked, choice, silhouette)
        masked = self._keep_masked(confidence, masked, math.cos(math.pi/2*(step+1)/total), tau)
        return torch.where(masked, 2, candidate), masked

    def start_glyphs(self, silhouette):
        """Glyph stage start: patches outside the silhouette are empty, inside masked."""
        b, n = silhouette.shape
        glyphs = torch.full((b, n, self.cells), self.mask_glyph, dtype=torch.long, device=silhouette.device)
        colors = torch.full_like(glyphs, self.mask_color)
        inside = silhouette == 1
        glyphs[~inside] = 0
        colors[~inside] = self.contract.empty_color
        return glyphs, colors, inside.clone()

    @torch.no_grad()
    def glyph_step(self, text, glyphs, colors, silhouette, masked, step, total):
        """One MaskGIT step of the glyph stage; returns (glyphs, colors, masked)."""
        gl, cl = self._guided(self.glyph_logits, text, glyphs, colors, silhouette)
        tau = self.config.temperature*(1-step/total)
        noise = -torch.log(-torch.log(torch.rand(gl.shape, device=gl.device).clamp(1e-6, 1-1e-6)))
        sampled_g = (gl+tau*noise).argmax(-1)
        sampled_c = cl.argmax(-1)
        confidence = F.log_softmax(gl, -1).gather(-1, sampled_g[..., None])[..., 0].mean(-1)
        still = self._keep_masked(confidence, masked, math.cos(math.pi/2*(step+1)/total), tau)
        fill = masked & ~still
        return (torch.where(fill[..., None], sampled_g, glyphs),
                torch.where(fill[..., None], sampled_c, colors), still)

    @torch.no_grad()
    def sample(self, text, steps=None, generator=None, return_trace=False):
        """text [B,T,768]. Returns glyphs, colors [B,R,C] (and per-step grids)."""
        c = self.config
        b, n, device = text.shape[0], self.ph*self.pw, text.device
        trace = []
        s_steps, g_steps = steps or (c.silhouette_steps, c.glyph_steps)
        silhouette = torch.full((b, n), 2, dtype=torch.long, device=device)
        masked = torch.ones((b, n), dtype=torch.bool, device=device)
        for step in range(s_steps):
            silhouette, masked = self.silhouette_step(text, silhouette, masked, step, s_steps)
            if return_trace:
                trace.append(self._render(silhouette, None, None))
        glyphs, colors, masked = self.start_glyphs(silhouette)
        for step in range(g_steps):
            glyphs, colors, masked = self.glyph_step(text, glyphs, colors, silhouette, masked, step, g_steps)
            if return_trace:
                trace.append(self._render(silhouette, glyphs, colors))
        g, col = self._render(silhouette, glyphs, colors)
        return (g, col, trace) if return_trace else (g, col)

    def _render(self, silhouette, glyphs, colors):
        """Displayable tokens at any stage: undecided silhouette patches stay empty."""
        b = silhouette.shape[0]
        if glyphs is None:
            # During the silhouette stage, show occupied patches as a light fill.
            fill = torch.full((b, silhouette.shape[1], self.cells), self.contract.glyphs.index(":"), device=silhouette.device)
            g = torch.where((silhouette == 1)[..., None], fill, 0)
            col = torch.full_like(g, self.contract.empty_color)
            col[g != 0] = len(self.contract.palette)-1
            return self.grid(g), self.grid(col)
        g = torch.where(glyphs == self.mask_glyph, 0, glyphs)
        col = torch.where(g == 0, self.contract.empty_color, colors.clamp(max=self.color_count-1))
        return self.grid(g), self.grid(col)


def cascade_loss(model, glyphs, colors, text_context, generator=None):
    """Both stages on one batch of real target grids [B,R,C]."""
    device = glyphs.device
    b = glyphs.shape[0]
    silhouette = model.silhouette_of(glyphs)
    # Stage A: masked silhouette with ratio from the cosine schedule.
    ratio = torch.cos(torch.rand(b, device=device)*math.pi/2).clamp(min=1/silhouette.shape[1])
    mask_a = torch.rand(silhouette.shape, device=device) < ratio[:, None]
    logits_a = model.silhouette_logits(torch.where(mask_a, 2, silhouette), text_context)
    loss_a = F.cross_entropy(logits_a[mask_a].float(), silhouette[mask_a])
    # Stage B: glyphs/colors inside the true silhouette, masked by patch.
    gp, cp = model.patches(glyphs), model.patches(colors)
    inside = silhouette == 1
    ratio = torch.cos(torch.rand(b, device=device)*math.pi/2).clamp(min=.05)
    mask_b = (torch.rand(silhouette.shape, device=device) < ratio[:, None]) & inside
    gin = torch.where(mask_b[..., None], model.mask_glyph, gp)
    cin = torch.where(mask_b[..., None], model.mask_color, cp)
    gl, cl = model.glyph_logits(gin, cin, silhouette, text_context)
    target_g = gp[mask_b]
    loss_g = F.cross_entropy(gl[mask_b].float().reshape(-1, model.glyph_count), target_g.reshape(-1)) if mask_b.any() else gl.sum()*0
    occupied = mask_b[..., None] & (gp != 0)
    loss_c = F.cross_entropy(cl[occupied].float(), cp[occupied].long()) if occupied.any() else cl.sum()*0
    return loss_a+loss_g+loss_c, {"silhouette_ce": float(loss_a.detach()), "glyph_ce": float(loss_g.detach()), "color_ce": float(loss_c.detach())}
