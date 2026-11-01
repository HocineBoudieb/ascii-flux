"""Real-data glyph diffusion experiment; auxiliary x0 CE, no pixel loss."""
import copy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from .causal_training import prepare
from .contract import TEXT_SPEC
from .discrete import DiscreteConfig,DiscreteGlyphFlow,replace_noise
from .masked import masked_loss
from .masked_training import MaskedTrainConfig,subject_groups,token_batch
from .storage import atomic_file,write_json
from .training import Corpus,restore_rng,rng_state


def inputs_for(model,batch,validation=False):
    g,c,pg,pc,text,mode = prepare(model,batch,validation)
    t = torch.rand(len(g),device=g.device)*.978+.02
    alpha = torch.cos(t*math.pi/2).square()
    noisy = replace_noise(g,alpha[:,None,None],len(model.contract.glyphs))
    context = (torch.rand_like(g.float()) < .1) & (mode[:,None,None] == 1)
    pg = pg.masked_fill(context,model.mask_glyph)
    pc = pc.masked_fill(context,model.mask_color)
    return noisy,torch.full_like(c,model.mask_color),pg,pc,text,1-alpha,mode


@torch.inference_mode()
def validate(model,corpus,groups,config,device):
    saved = rng_state();model.eval()
    try:
        torch.manual_seed(config.seed+1);rng = random.Random(config.seed+1)
        losses = []
        for _ in range(config.val_batches):
            batch = token_batch(corpus,groups,config,'val',device,rng)
            inputs = inputs_for(model,batch,True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=str(device).startswith('cuda')):
                gl,cl = model(*inputs)
            losses.append(float(masked_loss(gl,cl,batch[0],batch[1],torch.ones_like(batch[0],dtype=torch.bool))))
        return float(np.mean(losses))
    finally:
        restore_rng(saved);model.train()


def train_discrete(manifest,embeddings,reviews,output,config=None,device='cuda',resume=None,warm_start=None):
    config = config or MaskedTrainConfig()
    if min(config.steps,config.batch_size,config.validate_every,config.val_batches) <= 0 or config.lr <= 0 or config.memorize_one_frame:
        raise ValueError('Positive real-data experiment configuration required')
    corpus = Corpus(manifest,embeddings,reviews)
    groups = subject_groups(corpus,config.subjects)
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'latest.pt').exists() and not resume:raise FileExistsError('Resume or choose another diffusion run')
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
    model_config = DiscreteConfig()
    model = DiscreteGlyphFlow(corpus.contract,model_config).to(device)
    initialization = None
    if warm_start:
        data = Path(warm_start).read_bytes()
        ck = torch.load(warm_start,map_location='cpu',weights_only=True)
        corpus.contract.require(ck['contract'])
        if (ck['format'] != 'ascii-flux-v5-masked-1' or ck['provenance']['dataset_fingerprint'] != corpus.manifest['fingerprint']
                or ck['provenance']['training_concepts'] != list(config.subjects) or ck['provenance']['synthetic_encoder']):
            raise ValueError('Warm-start data/concept provenance mismatch')
        model.load_state_dict(ck['model'])
        initialization = {'checkpoint_sha256':hashlib.sha256(data).hexdigest(),'step':ck['step'],
                          'format':ck['format'],'optimizer_restarted':True}
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.lr,weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda',enabled=str(device).startswith('cuda'),init_scale=128.)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    code = hashlib.sha256()
    for name in ('discrete.py','discrete_training.py','masked.py','masked_training.py','causal_training.py','text.py'):
        code.update(Path(__file__).with_name(name).read_bytes())
    provenance = {'dataset_fingerprint':corpus.manifest['fingerprint'],'text_spec':TEXT_SPEC,
        'allow_fixture':False,'allow_unreviewed':False,'synthetic_encoder':False,'development_only':True,
        'recognition_validated':False,'training_concepts':list(config.subjects),'source_conditions':corpus.manifest['source'],
        'review_scope':corpus.review_scope,'training_code_sha256':code.hexdigest(),'torch_version':str(torch.__version__),
        'scope':'six-concept uniform glyph diffusion; recognition and motion unvalidated','initialization':initialization}
    start,best,best_step,best_model,history,skipped = 0,math.inf,0,None,[],0
    if resume:
        ck = torch.load(resume,map_location='cpu',weights_only=False);corpus.contract.require(ck['contract'])
        old,new = dict(ck['train_config']),asdict(config);old.pop('steps');new.pop('steps')
        if ck['provenance'] != provenance or old != new or ck['model_config'] != asdict(model_config):
            raise ValueError('Diffusion resume provenance/config mismatch')
        model.load_state_dict(ck['model']);ema.load_state_dict(ck['ema'])
        optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        start,best,best_step,best_model,history,skipped = (ck[k] for k in ('step','best_val','best_step','best_model','history','skipped_amp_updates'))
        restore_rng(ck['rng'])
    started = time.monotonic()
    print({'parameters':sum(p.numel() for p in model.parameters()),'scope':provenance['scope']},flush=True)
    for step in range(start,config.steps):
        batch = token_batch(corpus,groups,config,'train',device,random)
        inputs = inputs_for(model,batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16,enabled=str(device).startswith('cuda')):
            gl,cl = model(*inputs)
            loss = masked_loss(gl,cl,batch[0],batch[1],torch.ones_like(batch[0],dtype=torch.bool))
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite diffusion CE')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=not scaler.is_enabled())
        scaler.step(optimizer);scaler.update()
        if math.isfinite(float(norm)):
            decay = min(.995,(1+step)/(10+step))
            with torch.no_grad():
                for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,1-decay)
        else:skipped += 1
        if (step+1)%25 == 0:
            write_json(output/'progress.json',{'step':step+1,'total_steps':config.steps,'loss':float(loss.detach()),
                'elapsed_seconds':time.monotonic()-started,'optimizer_updates':step+1-skipped,'recognition_validated':False})
        if (step+1)%config.validate_every == 0 or step+1 == config.steps:
            value = validate(ema,corpus,groups,config,device)
            if not math.isfinite(value):raise FloatingPointError('Non-finite diffusion validation')
            if value < best:
                best,best_step = value,step+1
                best_model = copy.deepcopy({k:v.detach().cpu() for k,v in ema.state_dict().items()})
            row = {'step':step+1,'train_x0_ce':float(loss.detach()),'val_x0_ce':value,'best_val_ce':best,
                   'optimizer_updates':step+1-skipped};history.append(row)
            ck = {'format':'ascii-flux-v5-discrete-1','contract':corpus.contract.to_dict(),'model_config':asdict(model_config),
                'train_config':asdict(config),'provenance':provenance,'model':model.state_dict(),'ema':ema.state_dict(),
                'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'step':step+1,'best_val':best,'best_step':best_step,
                'best_model':best_model,'rng':rng_state(),'history':history,'skipped_amp_updates':skipped}
            with atomic_file(output/'latest.pt') as stream:torch.save(ck,stream)
            with atomic_file(output/'best.pt') as stream:
                torch.save({k:ck[k] for k in ('format','contract','model_config','provenance')} | {'model':best_model,'step':best_step},stream)
            write_json(output/'history.json',history);print(row,flush=True)
    return model
