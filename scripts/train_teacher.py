import os, yaml, torch, torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from models.teacher_darswin_unet import build_teacher_smp_unet_swin_tiny
from models.rdc import wrap_decoder_with_rdc
from utils.train_utils import set_seed, make_loaders
from utils.losses import ComboLoss

def evaluate(model, loader, device):
    model.eval()
    inter=0; union=0; correct=0; total=0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs=imgs.to(device); masks=masks.to(device)
            logits = model(imgs)
            preds = (torch.sigmoid(logits[:,1:2])>0.5).long().squeeze(1)
            correct += (preds==masks).sum().item()
            total += masks.numel()
            p1 = (preds==1); g1=(masks==1)
            inter += (p1 & g1).sum().item()
            union += (p1 | g1).sum().item()
    pix_acc = correct/total if total>0 else 0
    iou = inter/union if union>0 else 0
    return pix_acc, iou

def main():
    cfg = yaml.safe_load(open('configs/config.yaml','r'))
    set_seed(cfg['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader = make_loaders(cfg, is_teacher=True)
    model = build_teacher_smp_unet_swin_tiny(
        num_classes=cfg['num_classes'],
        encoder_name=str(cfg['teacher']['encoder_name']),
        encoder_weights=str(cfg['teacher']['encoder_weights'])
    )
    if cfg['rdc']['enable']:
        model.decoder = wrap_decoder_with_rdc(model.decoder, replace_stages=tuple(cfg['rdc']['replace_stages']))

    model = model.to(device)
    criterion = ComboLoss(w_bce=1.0, w_dice=1.0, w_boundary=0.5).to(device)
    optim = AdamW(model.parameters(), lr=float(cfg['teacher']['lr']))
    use_amp = bool(cfg['mixed_precision'])
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    best_iou=0.0
    os.makedirs(cfg['save_dir'], exist_ok=True)

    for ep in range(1, int(cfg['teacher']['epochs'])+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Teacher Epoch {ep}")
        for imgs, masks in pbar:
            imgs=imgs.to(device); masks=masks.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(imgs)            # 传原始 logits
                loss   = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            pbar.set_postfix(loss=float(loss))

        pix_acc, miou = evaluate(model, val_loader, device)
        print(f"[Teacher] Epoch {ep} val pixAcc={pix_acc:.4f} mIoU={miou:.4f}")
        if miou >= best_iou:
            best_iou=miou
            torch.save(model.state_dict(), os.path.join(cfg['save_dir'], 'teacher_swin_unet_best.pth'))
            print("  Saved best teacher.")

if __name__=='__main__':
    main()
