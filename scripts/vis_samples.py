# scripts/vis_samples.py
import os, yaml, cv2, torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# 项目内模块
from models.teacher_darswin_unet import build_teacher_smp_unet_swin_tiny
from models.rdc import wrap_decoder_with_rdc
from data.fisheye_json_dataset import FisheyeJsonDataset

def tensor_to_img_uint8(t):
    """(C,H,W) torch / float[0,1] -> (H,W,3) uint8 BGR for cv2"""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().float().clamp(0, 1).numpy()
    if t.ndim == 3 and t.shape[0] in (1,3):
        t = np.transpose(t, (1,2,0))
    if t.shape[2] == 1:
        t = np.repeat(t, 3, axis=2)
    img = (t * 255.0 + 0.5).astype(np.uint8)
    # 训练中读的是 BGR，这里保持 BGR 输出，避免颜色颠倒
    return img

def colorize_mask(mask, color=(0, 255, 0)):
    """(H,W) 0/1 -> (H,W,3) uint8; color 为 BGR"""
    h,w = mask.shape
    out = np.zeros((h,w,3), dtype=np.uint8)
    out[mask>0] = color
    return out

def overlay(img_bgr, mask, color=(0,255,0), alpha=0.35):
    """在 BGR 图上叠加 0/1 掩码颜色"""
    color_mask = colorize_mask(mask, color=color)
    return cv2.addWeighted(img_bgr, 1.0, color_mask, alpha, 0)

def put_label(img, text):
    """在左上角打标题"""
    img = img.copy()
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    return img

def vis_one(model, image_t, mask_t, device, thr=0.5):
    """对一张 (C,H,W) 图像可视化"""
    model.eval()
    with torch.no_grad():
        logits = model(image_t.unsqueeze(0).to(device))  # (1,2,H,W)
        prob = torch.softmax(logits, dim=1)[:, 1]        # 天空=通道1
        pred = (prob >= thr).float().cpu().squeeze(0).numpy().astype(np.uint8)

    # 准备四联图
    img = tensor_to_img_uint8(image_t)                   # (H,W,3) BGR
    gt  = mask_t.cpu().numpy().astype(np.uint8)          # (H,W), 0/1

    vis0 = put_label(img, 'Image')
    vis1 = put_label(colorize_mask(gt,  (0,255,0)), 'GT (sky=green)')
    vis2 = put_label(colorize_mask(pred,(0,0,255)), 'Pred (sky=red)')
    vis3 = put_label(overlay(img, gt,(0,255,0),0.35), 'Overlay GT') 
    vis4 = put_label(overlay(img, pred,(0,0,255),0.35), 'Overlay Pred')

    top = np.hstack([vis0, vis1, vis2])
    bot = np.hstack([vis3, vis4, np.zeros_like(vis4)])   # 占位补齐为 3 列
    grid = np.vstack([top, bot])
    return grid, pred

def load_model_from_cfg(cfg, device):
    model = build_teacher_smp_unet_swin_tiny(
        num_classes=cfg['num_classes'],
        encoder_name=str(cfg['teacher']['encoder_name']),
        encoder_weights=str(cfg['teacher'].get('encoder_weights') or 'None') if cfg['teacher'].get('encoder_weights', None) else None
    )
    if cfg.get('rdc', {}).get('enable', False):
        model.decoder = wrap_decoder_with_rdc(model.decoder, replace_stages=tuple(cfg['rdc']['replace_stages']))
    ckpt = os.path.join(cfg['save_dir'], 'teacher_swin_unet_best.pth')
    assert os.path.isfile(ckpt), f"找不到权重: {ckpt}"
    state = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(state, strict=True)
    return model.to(device)

def main(
    cfg_path='configs/config.yaml',
    list_path=None,
    out_dir='runs_vis',
    num_samples=12,
    seed=42,
    thr=0.5
):
    os.makedirs(out_dir, exist_ok=True)
    cfg = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))

    if list_path is None:
        list_path = cfg['val_list']   # 默认看验证集
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset/Loader：直接用验证增强（LongestMaxSize+PadIfNeeded+Normalize）
    ds = FisheyeJsonDataset(list_path, img_size=cfg['img_size'], is_train=False, num_classes=cfg['num_classes'])
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    # 模型
    print('>> loading model...')
    model = load_model_from_cfg(cfg, device)
    print('>> model ready')

    # 采样可视化
    n = min(num_samples, len(ds))
    print(f'>> visualize {n} samples to: {out_dir}')
    for i, (img_t, mask_t) in enumerate(tqdm(dl, total=n, desc='Visualizing')):
        if i >= n: break
        img_t = img_t[0]     # (C,H,W)
        mask_t = mask_t[0]   # (H,W)

        grid, pred = vis_one(model, img_t, mask_t, device, thr=thr)

        # 保存
        save_path = os.path.join(out_dir, f'vis_{i:04d}.jpg')
        cv2.imwrite(save_path, grid)
    print('>> done.')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser('Visualize teacher predictions vs GT')
    p.add_argument('--cfg', type=str, default='configs/config.yaml')
    p.add_argument('--list', type=str, default=None, help='默认用 cfg[val_list]')
    p.add_argument('--out', type=str, default='runs_vis')
    p.add_argument('--num', type=int, default=12)
    p.add_argument('--thr', type=float, default=0.5, help='天空阈值(softmax后通道1)')
    args = p.parse_args()
    main(cfg_path=args.cfg, list_path=args.list, out_dir=args.out, num_samples=args.num, thr=args.thr)
