import os, yaml, torch, cv2
import numpy as np
from tqdm import tqdm
from models.teacher_darswin_unet import build_teacher_smp_unet_swin_tiny
from models.rdc import wrap_decoder_with_rdc
from data.fisheye_json_dataset import FisheyeJsonDataset
from torch.utils.data import DataLoader

def main():
    cfg = yaml.safe_load(open('configs/config.yaml','r'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = build_teacher_smp_unet_swin_tiny(
        num_classes=cfg['num_classes'],
        encoder_name=str(cfg['teacher']['encoder_name']),
        encoder_weights=None
    )
    if cfg['rdc']['enable']:
        model.decoder = wrap_decoder_with_rdc(model.decoder, replace_stages=tuple(cfg['rdc']['replace_stages']))
    ckpt = os.path.join(cfg['save_dir'], 'teacher_swin_unet_best.pth')
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.to(device).eval()

    unlabeled_list = cfg.get('unlabeled_list', cfg['val_list'])
    ds = FisheyeJsonDataset(unlabeled_list, img_size=cfg['img_size'], is_train=False, num_classes=cfg['num_classes'])
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    os.makedirs(cfg['pseudo_dir'], exist_ok=True)

    for i,(img, _) in enumerate(tqdm(loader, desc="Pseudo-labeling")):
        img = img.to(device)
        with torch.no_grad():
            logits = model(img)
            prob  = torch.softmax(logits, dim=1)[:,1].squeeze(0).cpu().numpy()  # 天空通道
        np.save(os.path.join(cfg['pseudo_dir'], f"prob_{i:06d}.npy"), prob)
        mask = (prob>=0.5).astype(np.uint8)*255
        cv2.imwrite(os.path.join(cfg['pseudo_dir'], f"mask_{i:06d}.png"), mask)

    print("Pseudo labels saved to", cfg['pseudo_dir'])

if __name__=='__main__':
    main()
