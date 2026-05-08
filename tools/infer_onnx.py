# infer_onnx.py
import os, sys, cv2, numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    print("请先安装 onnxruntime 或 onnxruntime-gpu：pip install onnxruntime")
    sys.exit(1)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_session(onnx_path, use_gpu=True):
    providers = []
    if use_gpu:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    so = ort.SessionOptions()
    # 可选：图优化级别
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    iname = sess.get_inputs()[0].name
    oname = sess.get_outputs()[0].name
    return sess, iname, oname

def to_chw_tensor(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(img, (0,1,2))  # 这行仅为提示；下一行会换维
    chw = np.transpose(img, (2,0,1))  # HWC -> CHW
    return chw

def pad_to_stride(x, stride=32, pad_val=0.0):
    # x: CHW
    _, h, w = x.shape
    H = (h + stride - 1) // stride * stride
    W = (w + stride - 1) // stride * stride
    if H == h and W == w:
        return x, (0,0,0,0)
    pad = np.full((x.shape[0], H, W), pad_val, dtype=x.dtype)
    pad[:, :h, :w] = x
    return pad, (h, w, H, W)

def postprocess(logits, orig_h, orig_w):
    # logits: (1, C=2, H, W)
    prob = softmax(logits[0])  # (2, H, W)
    sky_prob = prob[1]         # 取第2通道作为天空概率
    # 裁回原始大小
    sky_prob = sky_prob[:orig_h, :orig_w]
    mask = (sky_prob >= 0.5).astype(np.uint8) * 255
    return sky_prob, mask

def softmax(x):
    x = x - np.max(x, axis=0, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=0, keepdims=True)

def colorize_heatmap(prob):
    # prob: [H,W] in [0,1]
    p8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(p8, cv2.COLORMAP_JET)

def overlay_mask(img_bgr, mask, alpha=0.5, color=(0,0,255)):
    # 把天空区域染成红色 (BGR)
    color_layer = np.zeros_like(img_bgr); color_layer[mask>0] = color
    return cv2.addWeighted(img_bgr, 1.0, color_layer, alpha, 0)

def predict_image(sess, iname, oname, img_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    h0, w0 = img_bgr.shape[:2]

    x = to_chw_tensor(img_bgr)                     # (C,H,W)
    x, (h, w, H, W) = pad_to_stride(x, stride=32)  # pad 到 32 的倍数
    x = x[None, ...].astype(np.float32)            # (1,C,H,W)

    logits = sess.run([oname], {iname: x})[0]      # (1,2,H,W)
    sky_prob, mask = postprocess(logits, h, w)

    # 可视化与保存
    base = Path(img_path).stem
    out_mask = os.path.join(out_dir, base + "_mask.png")
    out_overlay = os.path.join(out_dir, base + "_overlay.png")
    out_prob = os.path.join(out_dir, base + "_prob.png")

    cv2.imwrite(out_mask, mask)
    overlay = overlay_mask(img_bgr, mask)
    cv2.imwrite(out_overlay, overlay)
    cv2.imwrite(out_prob, colorize_heatmap(sky_prob))

    print(f"✓ {img_path} -> {out_mask}, {out_overlay}, {out_prob}")

def predict_folder(sess, iname, oname, img_dir, out_dir, exts=(".jpg",".png",".jpeg",".bmp")):
    imgs = [str(p) for p in Path(img_dir).rglob("*") if p.suffix.lower() in exts]
    if not imgs:
        print("未在该目录下找到图片：", img_dir); return
    for p in imgs:
        predict_image(sess, iname, oname, p, out_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="./checkpoints/segformer_b0_student.onnx", help="ONNX 模型路径")
    ap.add_argument("--image", help="单张图片路径")
    ap.add_argument("--dir", help="图片文件夹路径（与 --image 二选一）")
    ap.add_argument("--out", default="./runs_onnx", help="输出目录")
    ap.add_argument("--cpu", action="store_true", help="强制用 CPU（不尝试 GPU）")
    args = ap.parse_args()

    sess, iname, oname = load_session(args.onnx, use_gpu=not args.cpu)
    if args.image:
        predict_image(sess, iname, oname, args.image, args.out)
    elif args.dir:
        predict_folder(sess, iname, oname, args.dir, args.out)
    else:
        ap.error("请提供 --image 或 --dir 之一")
