"""
Evaluate a checkpoint. Edit the CONFIG block, then run:
  python eval_checkpoint.py
"""
import os, sys
os.chdir('/system/user/studentwork/wehofer/pino-lightning')
sys.path.insert(0, '.')

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from neuralop import LpLoss
from src.models.darcy_module import DarcyLitModule

# ── CONFIG ────────────────────────────────────────────────────────────────────
RUN_ID    = 'zd1kdvom'
CKPT      = f'lowres-binary-paper/{RUN_ID}/checkpoints/best.ckpt'
HYDRA_CFG = 'outputs/2026-03-27/19-48-27/.hydra/config.yaml'
DATA_ROOT = '/system/user/studentwork/wehofer/data/darcy_binary'
GPU       = 1                                        # GPU index to use
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device(f'cuda:{GPU}')
print(f'device: {device} ({torch.cuda.get_device_name(GPU)})')

cfg = OmegaConf.load(HYDRA_CFG)
OmegaConf.update(cfg, 'data.data_root', DATA_ROOT)

dm = instantiate(cfg.data)
dm.setup('test')

model = DarcyLitModule(config=cfg, data_processor=dm.data_processor)
model.eval()
model.to(device)
dm.data_processor.to(device)

raw = torch.load(CKPT, map_location='cpu', weights_only=False)
missing, unexpected = model.load_state_dict(raw['state_dict'], strict=False)
if missing:     print('MISSING:', missing)
if unexpected:  print('UNEXPECTED:', unexpected)

cb = next(iter(raw['callbacks'].values()))
print(f'epoch={raw["epoch"]}  monitor_score={float(cb["best_model_score"])*100:.4f}%')

lp = LpLoss(d=2, p=2, reduction='mean')
with torch.no_grad():
    for res, loader in sorted(dm._test_loaders.items()):
        num, denom = 0.0, 0
        for batch in loader:
            data  = dm.data_processor.preprocess(batch)
            preds = model.model(data['x'])
            preds = dm.data_processor.postprocess(preds)
            if model._bc_mollifier is not None:
                mol   = DarcyLitModule._build_mollifier(preds.shape[-1]).to(device)
                preds = preds * (model._mollifier_scale * mol)
            num   += lp(preds, data['y']).item() * data['y'].shape[0]
            denom += data['y'].shape[0]
        print(f'val_{res}_l2 = {num/denom*100:.4f}%')
