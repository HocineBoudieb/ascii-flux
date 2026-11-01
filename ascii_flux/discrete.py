"""Uniform categorical diffusion on glyph IDs; colors are a separate head.

Uses the discrete posterior parameterization of Austin et al. (D3PM, 2021,
https://arxiv.org/abs/2107.03006), not their architecture or reported results.
The glyph CE is an x0 denoising objective; no pixel/codec loss is used.
"""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F
from .masked import MaskedConfig,MaskedGlyphFlow


@dataclass(frozen=True)
class DiscreteConfig(MaskedConfig):
    emergence_steps: int = 32
    motion_steps: int = 8
    guidance: float = 2.
    temperature: float = 1.


def replace_noise(glyphs,retention,vocabulary):
    keep = torch.rand_like(glyphs.float()) < retention
    return torch.where(keep,glyphs,torch.randint(vocabulary,glyphs.shape,device=glyphs.device))


def reverse_probabilities(p0,current,alpha_current,alpha_next):
    """Exact mixture of q(x_s | x_t,x0) for uniform forward corruption.

    p0 is the model distribution over clean tokens conditional on x_t.
    alpha values are cumulative retention, not a per-step beta.
    """
    if not 0 <= alpha_current <= alpha_next <= 1 or alpha_current == 1:
        raise ValueError("Reverse diffusion requires 0 <= alpha_t <= alpha_s <= 1, alpha_t < 1")
    k = p0.shape[1]
    observed = F.one_hot(current.long(),k).permute(0,3,1,2).float()
    denominator = (1-alpha_current)/k + alpha_current*observed
    weights = p0.float()/denominator.clamp_min(1e-12)
    prior = alpha_next*weights+(1-alpha_next)/k*weights.sum(1,keepdim=True)
    transition = alpha_current/alpha_next if alpha_next else 0.
    likelihood = (1-transition)/k+transition*observed
    posterior = (prior*likelihood).clamp_min(0.)
    return posterior/posterior.sum(1,keepdim=True).clamp_min(1e-12)


class DiscreteGlyphFlow(MaskedGlyphFlow):
    """Hidden noisy glyphs + displayed glyph/color grid + previous native grid."""
    def __init__(self,contract,config=None):
        super().__init__(contract,config or DiscreteConfig())

    @staticmethod
    def retention(mode,phase,total):
        base = .65 if mode == 1 else .05 if mode == 2 else .0001
        return base+(1-base)*math.sin(phase/total*math.pi/2)**2

    def initial_state(self,batch=1,noisy=False):
        state = self.coordinates.new_zeros((batch,7,self.contract.rows,self.contract.cols))
        state[:,0] = torch.randint(len(self.contract.glyphs),state[:,0].shape,device=state.device)
        state[:,2] = self.contract.empty_color
        state[:,3] = self.mask_glyph;state[:,4] = self.mask_color
        return state

    def tokens(self,state):
        g,c = state[:,1].long(),state[:,2].long()
        return g,torch.where(g == 0,self.contract.empty_color,c)

    def recondition(self,state):
        result = state.clone()
        result[:,3:5] = state[:,1:3]
        result[:,0] = replace_noise(state[:,1].long(),self.retention(2,0,self.config.emergence_steps),len(self.contract.glyphs))
        result[:,5] = 0;result[:,6] = 2
        return result

    def step(self,state,text):
        state = state.clone()
        phase,mode = int(state[0,5,0,0]),int(state[0,6,0,0])
        total = self.config.motion_steps if mode == 1 else self.config.emergence_steps
        if phase >= total:
            state[:,3:5] = state[:,1:3]
            state[:,0] = replace_noise(state[:,1].long(),self.retention(1,0,self.config.motion_steps),len(self.contract.glyphs))
            state[:,5] = 0;state[:,6] = 1
            phase,mode,total = 0,1,self.config.motion_steps
        a,b = self.retention(mode,phase,total),self.retention(mode,phase+1,total)
        g = state[:,0].long();c = torch.full_like(g,self.mask_color)
        pg,pc = state[:,3].long(),state[:,4].long()
        ratio = text.new_full((len(state),),1-a)
        modes = torch.full((len(state),),mode,device=state.device,dtype=torch.long)
        if self.config.guidance != 1:
            gl,cl = self(torch.cat([g,g]),torch.cat([c,c]),torch.cat([pg,pg]),torch.cat([pc,pc]),
                         torch.cat([text,torch.zeros_like(text)]),torch.cat([ratio,ratio]),torch.cat([modes,modes]))
            gl,ugl = gl.chunk(2);cl,ucl = cl.chunk(2)
            gl = ugl+self.config.guidance*(gl-ugl);cl = ucl+self.config.guidance*(cl-ucl)
        else:gl,cl = self(g,c,pg,pc,text,ratio,modes)
        p0 = F.softmax(gl.float()/self.config.temperature,1)
        posterior = reverse_probabilities(p0,g,a,b)
        random = torch.rand_like(posterior).clamp_(1e-6,1-1e-6)
        state[:,0] = (posterior.clamp_min(1e-12).log()-torch.log(-torch.log(random))).argmax(1)
        # The displayed grid is the model's clean-token readout, while noisy
        # categorical tokens remain hidden. No raster or glyph converter.
        display = gl.argmax(1)
        state[:,1] = display
        state[:,2] = torch.where(display != 0,cl.argmax(1),self.contract.empty_color)
        state[:,5] = phase+1
        return state,gl,cl
