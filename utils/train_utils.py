import random, numpy as np, torch
from torch.utils.data import DataLoader
from data.fisheye_json_dataset import FisheyeJsonDataset

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_loaders(cfg, is_teacher=True):
    train_ds = FisheyeJsonDataset(cfg['train_list'], img_size=cfg['img_size'], is_train=True, num_classes=cfg['num_classes'])
    val_ds   = FisheyeJsonDataset(cfg['val_list'], img_size=cfg['img_size'], is_train=False, num_classes=cfg['num_classes'])
    train_loader = DataLoader(train_ds, batch_size=cfg['train_batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg['val_batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=True)
    return train_loader, val_loader
