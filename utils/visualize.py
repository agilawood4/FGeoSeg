# utils/visualize.py
import cv2, numpy as np

def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray, color=(0,0,255), alpha=0.5):
    lay = np.zeros_like(img_bgr); lay[mask>0] = color
    return cv2.addWeighted(img_bgr, 1.0, lay, alpha, 0.0)

def prob_to_heatmap(prob: np.ndarray):
    p8 = (np.clip(prob, 0, 1)*255).astype(np.uint8)
    return cv2.applyColorMap(p8, cv2.COLORMAP_JET)
