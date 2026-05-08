import os, yaml, torch, torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from models.student_segformer_b0 import SegFormerB0Sky
from utils.train_utils import set_seed, make_loaders

def upsample_to(logits, target_size):
    return F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)

def cps_loss(logits_a, logits_b):
    with torch.no_grad():
        prob_a = torch.softmax(logits_a, dim=1)
        prob_b = torch.softmax(logits_b, dim=1)
        pseudo_a = prob_a.argmax(dim=1)
        pseudo_b = prob_b.argmax(dim=1)
    loss = F.cross_entropy(logits_a, pseudo_b) + F.cross_entropy(logits_b, pseudo_a)
    return loss*0.5

def evaluate_simple(model, loader, device):
    model.eval()
    inter=0; union=0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs=imgs.to(device); masks=masks.to(device)
            out = model(imgs)
            logits = out.logits if hasattr(out,'logits') else out
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            preds = logits.argmax(dim=1)
            p1 = (preds==1); g1=(masks==1)
            inter += (p1 & g1).sum().item()
            union += (p1 | g1).sum().item()
    return (inter/union) if union>0 else 0.0

def main():
    cfg = yaml.safe_load(open('configs/config.yaml','r'))
    set_seed(cfg['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader = make_loaders(cfg, is_teacher=False)

    stu1 = SegFormerB0Sky(num_labels=cfg['num_classes']).to(device)
    stu2 = SegFormerB0Sky(num_labels=cfg['num_classes']).to(device)

    opt1 = AdamW(stu1.parameters(), lr=float(cfg['student']['lr']))
    opt2 = AdamW(stu2.parameters(), lr=float(cfg['student']['lr']))

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg['mixed_precision']))

    best_iou=0.0
    os.makedirs(cfg['save_dir'], exist_ok=True)

    for ep in range(1, int(cfg['student']['epochs'])+1):
        stu1.train(); stu2.train()
        pbar = tqdm(train_loader, desc=f"Student CPS Epoch {ep}")
        for imgs, masks in pbar:
            imgs=imgs.to(device); masks=masks.to(device)
            opt1.zero_grad(set_to_none=True); opt2.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(cfg['mixed_precision'])):
                out1 = stu1(imgs); out2 = stu2(imgs)
                logits1 = out1.logits if hasattr(out1,'logits') else out1
                logits2 = out2.logits if hasattr(out2,'logits') else out2
                if logits1.shape[-2:] != masks.shape[-2:]:
                    logits1 = upsample_to(logits1, masks.shape[-2:])
                    logits2 = upsample_to(logits2, masks.shape[-2:])

                sup_loss = F.cross_entropy(logits1, masks) + F.cross_entropy(logits2, masks)
                unsup_loss = cps_loss(logits1, logits2)
                loss = sup_loss + unsup_loss

            scaler.scale(loss).backward()
            scaler.step(opt1); scaler.step(opt2)
            scaler.update()
            pbar.set_postfix(loss=float(loss))

        miou = evaluate_simple(stu1, val_loader, device)
        print(f"[Student] Epoch {ep} val mIoU={miou:.4f}")
        if miou>=best_iou:
            best_iou=miou
            torch.save(stu1.state_dict(), os.path.join(cfg['save_dir'], 'student_segformer_b0_best.pth'))
            print("  Saved best student (stu1).")

if __name__=='__main__':
    main()
