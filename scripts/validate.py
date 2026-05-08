# scripts/validate.py
import os, sys, json, cv2, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import bin_stats, bin_metrics, reduce_metrics
from utils.visualize import overlay_mask
from data.fisheye_json_dataset import FisheyeJsonDataset  # 你已有的Dataset

IMAGENET_MEAN = np.array([0.485,0.456,0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229,0.224,0.225], dtype=np.float32)

# scripts/validate.py 顶部 imports 之后
def _rasterize_json_local(json_path, hw):
    import json, cv2, numpy as np
    H, W = hw
    mask = np.zeros((H, W), np.uint8)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def to_pts(lst):
        pts = np.array([[int(round(x)), int(round(y))] for x, y in lst], np.int32)
        if pts.ndim != 2 or pts.shape[0] < 3:
            return None
        return pts.reshape(-1, 1, 2)

    polys = []
    if isinstance(data.get("sky"), list):
        for poly in data["sky"]:
            p = to_pts(poly);  polys.append(p) if p is not None else None

    if isinstance(data.get("objects"), list):
        for obj in data["objects"]:
            if str(obj.get("label","")).lower() == "sky":
                p = to_pts(obj.get("polygon") or obj.get("points", []))
                polys.append(p) if p is not None else None

    if isinstance(data.get("polygons"), list):
        for obj in data["polygons"]:
            if str(obj.get("label","")).lower() == "sky":
                p = to_pts(obj.get("points") or obj.get("polygon", []))
                polys.append(p) if p is not None else None

    if polys:
        cv2.fillPoly(mask, polys, 255)
    return mask

def _preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    img = (img - IMAGENET_MEAN)/IMAGENET_STD
    chw = np.transpose(img, (2,0,1))[None].astype(np.float32)  # (1,3,H,W)
    return chw

def _pad32(x):  # (1,C,H,W)
    _,_,h,w = x.shape
    H = (h+31)//32*32; W = (w+31)//32*32
    if H==h and W==w: return x, (h,w)
    pad = np.zeros((1,x.shape[1],H,W), dtype=x.dtype)
    pad[:,:,:h,:w] = x
    return pad, (h,w)

def _softmax(logits):  # (1,2,H,W)
    x = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(x); return ex/ex.sum(axis=1, keepdims=True)

def _infer_onnx(sess, iname, oname, img_bgr):
    x = _preprocess(img_bgr); x, (h,w) = _pad32(x)
    logits = sess.run([oname], {iname: x})[0]  # (1,2,H,W)
    prob = _softmax(logits)[0,1,:h,:w]
    mask = (prob>=0.5).astype(np.uint8)*255
    return prob, mask

def _infer_torch(model, device, img_bgr):
    import torch
    with torch.no_grad():
        x = _preprocess(img_bgr)
        x, (h,w) = _pad32(x)
        xt = torch.from_numpy(x).to(device)
        logits = model(xt)  # (1,2,H,W)
        if isinstance(logits, (list,tuple)): logits = logits[0]
        prob = torch.softmax(logits, dim=1)[0,1,:h,:w].float().cpu().numpy()
        mask = (prob>=0.5).astype(np.uint8)*255
        return prob, mask

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_list", default="./dataset/val_list.txt")
    ap.add_argument("--save_dir", default="./runs_val")
    ap.add_argument("--onnx", help="若提供则用 ONNXRuntime 验证")
    ap.add_argument("--pth", help="若提供则用 PyTorch 学生权重验证")
    ap.add_argument("--cuda", action="store_true", default="cpu")
    ap.error = ap.error  # silence pyright
    args = ap.parse_args()

    try:
        raster = FisheyeJsonDataset.rasterize_json
    except AttributeError:
        raster = _rasterize_json_local

    os.makedirs(args.save_dir, exist_ok=True)
    # 载入验证清单
    with open(args.val_list, "r", encoding="utf-8") as f:
        pairs = [ln.strip().split("\t") for ln in f if ln.strip()]
    # 推理后端
    run_onnx = args.onnx is not None
    if run_onnx:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider","CPUExecutionProvider"] if args.cuda else ["CPUExecutionProvider"]
        sess = ort.InferenceSession(args.onnx, providers=providers)
        iname = sess.get_inputs()[0].name; oname = sess.get_outputs()[0].name
    else:
        import torch
        device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        # 载入你训练用的学生模型构造器
        from models.student_segformer_b0 import build_student_segformer_b0  # 需存在
        model = build_student_segformer_b0(num_classes=2)
        assert args.pth and os.path.isfile(args.pth), "请提供 --pth 学生权重路径"
        sd = torch.load(args.pth, map_location="cpu")
        model.load_state_dict(sd, strict=False); model.eval().to(device)

    metrics = []
    for i,(img_path, json_path) in enumerate(pairs, 1):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None: print("读图失败:", img_path); continue
        # 读 GT 掩码（沿用你的Dataset栅格化逻辑）
        gt = raster(json_path, img.shape[:2])  # 需要该静态方法；若无，可调用内部函数

        if run_onnx:
            prob, pred = _infer_onnx(sess, iname, oname, img)
        else:
            prob, pred = _infer_torch(model, device, img)
        
        # 这里之前出过 bug，需要保证 pred/prob 与 gt 同大小
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        tp,fp,fn,tn = bin_stats(pred, gt)
        m = bin_metrics(tp,fp,fn,tn)
        metrics.append(m)

        # 可选保存可视化
        overlay = overlay_mask(img, pred)
        base = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(args.save_dir, f"{base}_mask.png"), pred)
        cv2.imwrite(os.path.join(args.save_dir, f"{base}_overlay.png"), overlay)

        if i % 20 == 0:
            avg = reduce_metrics(metrics)
            print(f"[{i}/{len(pairs)}] mIoU={avg['mIoU']:.4f} Dice={avg['Dice']:.4f} Acc={avg['Acc']:.4f}")

    avg = reduce_metrics(metrics)
    print("==== Validation Result ====")
    for k,v in avg.items(): print(f"{k}: {v:.4f}")
    with open(os.path.join(args.save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(avg, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
