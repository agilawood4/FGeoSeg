# scripts/predict.py
import os, sys, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.visualize import overlay_mask, prob_to_heatmap

IMAGENET_MEAN = np.array([0.485,0.456,0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229,0.224,0.225], dtype=np.float32)

def _preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    img = (img - IMAGENET_MEAN)/IMAGENET_STD
    return np.transpose(img, (2,0,1))[None].astype(np.float32)

def _pad32(x):
    _,_,h,w = x.shape; H=(h+31)//32*32; W=(w+31)//32*32
    if H==h and W==w: return x,(h,w)
    pad = np.zeros((1,x.shape[1],H,W), dtype=x.dtype); pad[:,:,:h,:w]=x
    return pad,(h,w)

def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x); return ex/ex.sum(axis=1, keepdims=True)

def infer_onnx(onnx_path, img_paths, out_dir, use_gpu=True):
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider","CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    iname = sess.get_inputs()[0].name; oname = sess.get_outputs()[0].name
    os.makedirs(out_dir, exist_ok=True)

    for p in img_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR); h0,w0 = img.shape[:2]
        x = _preprocess(img); x,(h,w) = _pad32(x)
        logits = sess.run([oname], {iname:x})[0]
        prob = _softmax(logits)[0,1,:h,:w]
        mask = (prob>=0.5).astype(np.uint8)*255
        base = Path(p).stem
        cv2.imwrite(os.path.join(out_dir, base+"_mask.png"), mask)
        cv2.imwrite(os.path.join(out_dir, base+"_overlay.png"), overlay_mask(img, mask))
        cv2.imwrite(os.path.join(out_dir, base+"_prob.png"), prob_to_heatmap(prob))
        print("✓", p)

def infer_pth(pth_path, img_paths, out_dir, use_gpu=True):
    import torch
    from models.student_segformer_b0 import build_student_segformer_b0
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = build_student_segformer_b0(num_classes=2)
    sd = torch.load(pth_path, map_location="cpu"); model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for p in img_paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR); h0,w0 = img.shape[:2]
            x = _preprocess(img); x,(h,w) = _pad32(x)
            xt = torch.from_numpy(x).to(device)
            logits = model(xt)
            if isinstance(logits, (list,tuple)): logits = logits[0]
            prob = torch.softmax(logits, dim=1)[0,1,:h,:w].float().cpu().numpy()
            mask = (prob>=0.5).astype(np.uint8)*255
            base = Path(p).stem
            cv2.imwrite(os.path.join(out_dir, base+"_mask.png"), mask)
            cv2.imwrite(os.path.join(out_dir, base+"_overlay.png"), overlay_mask(img, mask))
            cv2.imwrite(os.path.join(out_dir, base+"_prob.png"), prob_to_heatmap(prob))
            print("✓", p)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="单张图片路径")
    ap.add_argument("--dir", help="图片文件夹路径（二选一）")
    ap.add_argument("--out", default="./runs_pred")
    ap.add_argument("--onnx", help="如果提供则用 ONNX 推理")
    ap.add_argument("--pth", help="否则使用 PyTorch 学生权重")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    imgs = []
    if args.image: imgs = [args.image]
    elif args.dir: imgs = [str(p) for p in Path(args.dir).rglob("*") if p.suffix.lower() in (".jpg",".png",".jpeg",".bmp")]
    else: ap.error("请提供 --image 或 --dir")

    if args.onnx:
        infer_onnx(args.onnx, imgs, args.out, use_gpu=not args.cpu)
    elif args.pth:
        infer_pth(args.pth, imgs, args.out, use_gpu=not args.cpu)
    else:
        ap.error("请提供 --onnx 或 --pth")

if __name__ == "__main__":
    main()
