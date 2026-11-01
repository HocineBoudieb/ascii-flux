"""Experimental causal transformer on native glyph/color patches.

Patches group existing cells for attention; no learned image codec, VAE, raster
intermediate or corpus retrieval is involved. Both outputs remain per-cell IDs.
"""
from dataclasses import asdict,dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CausalConfig:
    width: int = 192
    heads: int = 6
    layers: int = 6
    patch_size: int = 4
    patches_per_step: int = 4
    text_dim: int = 768
    text_dropout: float = .1
    temperature: float = .8
    guidance: float = 1.5

    def __post_init__(self):
        if min(self.width,self.heads,self.layers,self.patch_size,self.patches_per_step,self.text_dim) < 1 or self.width % self.heads:
            raise ValueError("Invalid causal transformer dimensions")
        if not 0 <= self.text_dropout < 1 or self.temperature <= 0 or self.guidance < 0:
            raise ValueError("Invalid causal sampling settings")


class CausalGlyphFlow(nn.Module):
    def __init__(self,contract,config=None):
        super().__init__()
        self.contract,self.config = contract,config or CausalConfig()
        cfg = self.config
        p,d = cfg.patch_size,cfg.width
        if contract.rows % p or contract.cols % p:
            raise ValueError("Grid must be divisible into complete native-cell patches")
        self.patch_count = contract.rows*contract.cols//p**2
        self.mask_glyph,self.mask_color = len(contract.glyphs),len(contract.palette)+1
        self.glyph_embedding = nn.Embedding(self.mask_glyph+1,16)
        self.color_embedding = nn.Embedding(self.mask_color+1,16)
        self.patch_projection = nn.Linear(p*p*32,d)
        self.previous_projection = nn.Linear(p*p*32,d)
        self.text_projection = nn.Sequential(nn.Linear(cfg.text_dim,d),nn.SiLU(),nn.Linear(d,d))
        self.position = nn.Parameter(torch.randn(1,self.patch_count,d)*.02)
        self.bos = nn.Parameter(torch.randn(1,1,d)*.02)
        self.mode_embedding = nn.Embedding(3,d)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d,cfg.heads,d*4,dropout=0.,
            activation="gelu",batch_first=True,norm_first=True),cfg.layers,norm=nn.LayerNorm(d))
        # Explicitly initialize distinct layers (TransformerDecoder clones its
        # template, whose identical initialization is an avoidable symmetry).
        for module in self.decoder.modules():
            if isinstance(module,nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:nn.init.zeros_(module.bias)
        self.glyph_head = nn.Linear(d,p*p*len(contract.glyphs))
        self.color_head = nn.Linear(d,p*p*len(contract.palette))
        self.register_buffer("causal_mask",torch.triu(torch.ones(self.patch_count,self.patch_count,dtype=torch.bool),1))

    def patches(self,value):
        b,r,c = value.shape[:3]
        p = self.config.patch_size
        tail = value.shape[3:]
        value = value.reshape(b,r//p,p,c//p,p,*tail)
        return value.permute(0,1,3,2,4,*range(5,value.ndim)).reshape(b,self.patch_count,p*p,*tail)

    def unpatch(self,value):
        b,n,a = value.shape[:3]
        p = self.config.patch_size
        tail = value.shape[3:]
        value = value.reshape(b,self.contract.rows//p,self.contract.cols//p,p,p,*tail)
        return value.permute(0,1,3,2,4,*range(5,value.ndim)).reshape(b,self.contract.rows,self.contract.cols,*tail)

    def embed(self,g,c):
        g = g.long()
        c = torch.where(g == self.mask_glyph,self.mask_color,
                        torch.where(g == 0,len(self.contract.palette),c)).long()
        return self.patches(torch.cat([self.glyph_embedding(g),self.color_embedding(c)],-1)).flatten(2)

    def forward(self,g,c,pg,pc,text,mode,limit=None):
        limit = self.patch_count if limit is None else limit
        condition = self.text_projection(text*math.sqrt(self.config.text_dim))+self.mode_embedding(mode.long())
        target = self.patch_projection(self.embed(g,c))[:,:max(0,limit-1)]
        target = torch.cat([self.bos.expand(len(g),-1,-1),target],1)
        target = target+self.position[:,:limit]+condition[:,None]
        memory = self.previous_projection(self.embed(pg,pc))+self.position
        memory = torch.cat([condition[:,None],memory],1)
        value = self.decoder(target,memory,tgt_mask=self.causal_mask[:limit,:limit])
        shape = (len(g),limit,self.config.patch_size**2)
        return (self.glyph_head(value).reshape(*shape,len(self.contract.glyphs)),
                self.color_head(value).reshape(*shape,len(self.contract.palette)))

    def initial_state(self,batch=1,noisy=False):
        state = self.position.new_zeros((batch,6,self.contract.rows,self.contract.cols))
        state[:,1] = self.contract.empty_color
        state[:,2] = self.mask_glyph
        state[:,3] = self.mask_color
        return state

    def recondition(self,state):
        result = state.clone()
        result[:,2:4] = result[:,:2]
        result[:,4] = 0
        result[:,5] = 2
        return result

    def tokens(self,state):
        g,c = state[:,0].long(),state[:,1].long()
        return g,torch.where(g == 0,self.contract.empty_color,c)

    def step(self,state,text):
        state = state.clone()
        cursor = int(state[0,4,0,0].item())
        mode = int(state[0,5,0,0].item())
        if cursor >= self.patch_count:
            state[:,2:4] = state[:,:2]
            state[:,4] = 0
            state[:,5] = 1
            cursor,mode = 0,1
        modes = torch.full((len(state),),mode,dtype=torch.long,device=state.device)
        for position in range(cursor,min(cursor+self.config.patches_per_step,self.patch_count)):
            g,c,pg,pc = (state[:,i].long() for i in range(4))
            if self.config.guidance != 1:
                gl,cl = self(torch.cat([g,g]),torch.cat([c,c]),torch.cat([pg,pg]),torch.cat([pc,pc]),
                    torch.cat([text,torch.zeros_like(text)]),torch.cat([modes,modes]),limit=position+1)
                gl,ugl = gl[:,-1].chunk(2);cl,ucl = cl[:,-1].chunk(2)
                gl = ugl+self.config.guidance*(gl-ugl);cl = ucl+self.config.guidance*(cl-ucl)
            else:
                gl,cl = self(g,c,pg,pc,text,modes,limit=position+1)
                gl,cl = gl[:,-1],cl[:,-1]
            temperature = self.config.temperature*(.5 if mode == 1 else 1.)
            sample_g = (gl.float()+temperature*(-torch.log(-torch.log(torch.rand_like(gl).clamp_(1e-6,1-1e-6))))).argmax(-1)
            sample_c = (cl.float()+temperature*(-torch.log(-torch.log(torch.rand_like(cl).clamp_(1e-6,1-1e-6))))).argmax(-1)
            sample_c = torch.where(sample_g != 0,sample_c,self.contract.empty_color)
            p = self.config.patch_size
            row,col = divmod(position,self.contract.cols//p)
            state[:,0,row*p:(row+1)*p,col*p:(col+1)*p] = sample_g.reshape(-1,p,p)
            state[:,1,row*p:(row+1)*p,col*p:(col+1)*p] = sample_c.reshape(-1,p,p)
        state[:,4] = min(cursor+self.config.patches_per_step,self.patch_count)
        return state,gl,cl

    def config_dict(self):return asdict(self.config)


def causal_loss(model,gl,cl,g,c):
    pg,pc = model.patches(g),model.patches(c)
    glyph = F.cross_entropy(gl.flatten(0,2).float(),pg.flatten().long())
    active = pg != 0
    color = F.cross_entropy(cl[active].float(),pc[active].long()) if active.any() else cl.sum()*0
    return glyph+color
