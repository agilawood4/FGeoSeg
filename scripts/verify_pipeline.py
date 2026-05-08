import os, json, random, yaml, cv2, numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# === 你的工程内模块 ===
from data.fisheye_json_dataset import FisheyeJsonDataset, load_polygon_mask
from models.teacher_darswin_unet import build_teacher_smp_unet_swin_tiny
from utils.losses import ComboLoss

def overlay_mask(img_bgr, mask01):
    lay = np.zeros_like(img_bgr); lay[mask01>0] = (0,0,255)
    return cv2.addWeighted(img_bgr, 1.0, lay, 0.5, 0)

def bin_stats(pred01, gt01):
    p = (pred01>0).astype(np.uint8); g = (gt01>0).astype(np.uint8)
    tp = int(((p==1)&(g==1)).sum()); fp = int(((p==1)&(g==0)).sum())
    fn = int(((p==0)&(g==1)).sum()); tn = int(((p==0)&(g==0)).sum())
    return tp,fp,fn,tn

def iou_from_stats(tp,fp,fn):
    return tp / (tp+fp+fn+1e-7)

def main():
    # 1) 读取配置
    cfg = yaml.safe_load(open('configs/config.yaml','r'))
    val_list = cfg.get('val_list', './dataset/val_list.txt')
    img_size = int(cfg.get('img_size', 512))
    os.makedirs('runs_debug', exist_ok=True)

    # 2) 数据集 / DataLoader
    ds = FisheyeJsonDataset(val_list, img_size=img_size, is_train=False, num_classes=2)
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    # 3) 抽样检查原始 mask 与增强后 mask 是否纯二值
    print('=== [Step A] 原始/增强掩码唯一值检查 ===')
    with open(val_list, 'r', encoding='utf-8') as f:
        pairs = [ln.strip().split('\t') for ln in f if ln.strip()]
    random.shuffle(pairs)

    for i,(img_p, ann_p) in enumerate(pairs[:3]):  # 检3张
        img = cv2.imread(img_p, cv2.IMREAD_COLOR)
        if img is None:
            print(f'  ❌ 读图失败: {img_p}')
            continue
        H,W = img.shape[:2]
        print(f'[{i}] img={os.path.basename(img_p)}')
        print(f'    anno={ann_p}  exists={os.path.isfile(ann_p)}  ext={os.path.splitext(ann_p)[1].lower()}')

    # 关键：统一用 load_polygon_mask，并打开 debug_print
    raw_mask = load_polygon_mask(H, W, ann_p, sky_label=1, debug_print=True)
    u_raw = np.unique(raw_mask)
    print(f'  >>>栅格化后 uniq={u_raw.tolist()}  fg_ratio={(raw_mask>0).mean():.4f}')

    # 取一批经过增强的数据观察唯一值
    images, masks = next(iter(loader))
    u_aug = torch.unique(masks).cpu().numpy().tolist()
    print('  augmented mask uniq (should be [0,1]):', u_aug)

    # 判定二值性
    if not set(u_aug).issubset({0,1}):
        print('!!! 警告：增强后的 mask 不是纯二值，请检查 A.Resize 的 mask_interpolation 是否为 cv2.INTER_NEAREST')
        return

    # 4) 构建模型（教师，仅用于自检）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_teacher_smp_unet_swin_tiny(
        num_classes=2,
        encoder_name=str(cfg['teacher']['encoder_name']),
        encoder_weights=str(cfg['teacher'].get('encoder_weights', None))
    ).to(device).eval()

    # 5) 前向一次：检查通道/概率/前景占比；并计算两种通道的 IoU 以验证“天空通道=1”
    print('=== [Step B] 前向/通道检查 ===')
    images = images.to(device); masks = masks.to(device)
    with torch.no_grad():
        logits = model(images)  # 原始 logits
        prob_ch1 = torch.softmax(logits, dim=1)[:,1]    # 天空通道=1
        prob_ch0 = torch.softmax(logits, dim=1)[:,0]
        pred1 = (prob_ch1>=0.5).long()
        pred0 = (prob_ch0>=0.5).long()

    gt_fg = float((masks>0).float().mean().item())
    p1_fg = float(pred1.float().mean().item())
    p0_fg = float(pred0.float().mean().item())
    print(f'  GT_fg={gt_fg:.4f}  Pred_fg(ch1)={p1_fg:.4f}  Pred_fg(ch0)={p0_fg:.4f}')

    # 6) 计算天空类 IoU（通道1/通道0 各算一次，用于确认哪路才对）
    tp1=fp1=fn1=0; tp0=fp0=fn0=0
    for b in range(masks.size(0)):
        g = masks[b].cpu().numpy().astype(np.uint8)
        p1 = pred1[b].cpu().numpy().astype(np.uint8)
        p0 = pred0[b].cpu().numpy().astype(np.uint8)
        a,b1,c,_ = bin_stats(p1, g); tp1 += a; fp1 += b1; fn1 += c
        a,b1,c,_ = bin_stats(p0, g); tp0 += a; fp0 += b1; fn0 += c

    iou1 = iou_from_stats(tp1,fp1,fn1)
    iou0 = iou_from_stats(tp0,fp0,fn0)
    print(f'  IoU(ch1-as-sky)={iou1:.4f}  IoU(ch0-as-sky)={iou0:.4f}')
    if iou0 > iou1:
        print('!!! 提示：通道0 的 IoU 更高，可能你的“天空通道”应为 ch0，请统一修改取通道的地方。')

    # 7) 跑一次 ComboLoss，确认不会触发 autocast 报错
    print('=== [Step C] Loss/AMP 检查 ===')
    use_amp = bool(cfg.get('mixed_precision', 0))
    criterion = ComboLoss().to(device)
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)
    images.requires_grad_(True)
    try:
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        print(f'  ComboLoss={float(loss):.4f}  (AMP enabled={use_amp}) ✓')
    except Exception as e:
        print('!!! Loss/AMP 出错：', e)
        return

    # 8) 保存三张可视化，人工快检
    print('=== [Step D] 可视化 ===')
    images_np = (images.detach().cpu().numpy().transpose(0,2,3,1) * 255).clip(0,255).astype(np.uint8)
    for b in range(min(3, masks.size(0))):
        img = images_np[b][:,:,::-1]  # 近似可视化（归一化后*255，不是精确反归一）
        gt  = masks[b].cpu().numpy().astype(np.uint8)
        pr  = pred1[b].cpu().numpy().astype(np.uint8)
        cv2.imwrite(f'runs_debug/sample{b}_gt.png', gt*255)
        cv2.imwrite(f'runs_debug/sample{b}_pred.png', pr*255)
        cv2.imwrite(f'runs_debug/sample{b}_overlay.png', overlay_mask(img, pr))
    print('  已保存到 runs_debug/ 目录。')

    # 9) 最后输出判定
    print('\n=== 自检结果总结 ===')
    ok_bin = set(u_aug).issubset({0,1})
    print(f'掩码是否纯二值(增强后)：{ok_bin}')
    print(f'前景占比：GT={gt_fg:.4f}  Pred(ch1)={p1_fg:.4f}')
    print(f'IoU(天空=ch1)={iou1:.4f}  vs  IoU(天空=ch0)={iou0:.4f}')
    if gt_fg == 0.0:
        print('>>> 你的 GT 前景比例为 0，请检查 JSON 栅格化/文件路径是否正确。')
    if p1_fg == 0.0:
        print('>>> 你的模型当前几乎全背景预测，可能仍在早期或通道/损失配置不对。先看 runs_debug/ 可视化。')

if __name__ == '__main__':
    main()
