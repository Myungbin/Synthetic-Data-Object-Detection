import torch

from archive.src.model.basemodel import build_model
from archive.src.config.config import cfg, seed_everything
from archive.src.train.training import train
from archive.src.evaluation.evaluation import inference
from archive.src.data.dataloader import DatasetLoader
from archive.src.train.scheduler import CosineAnnealingWarmUpRestarts

seed_everything(cfg.SEED)


dataloader = DatasetLoader()
train_loader, test_loader = dataloader.loader()
model = build_model()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=1, eta_max=0.1, T_up=10, gamma=0.5)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

infer_model = train(model, train_loader, optimizer, scheduler)
inference(infer_model, test_loader)



