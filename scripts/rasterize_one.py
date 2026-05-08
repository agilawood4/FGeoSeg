import cv2, numpy as np
from data.fisheye_json_dataset import load_polygon_mask

IMG = r"E:\fisheye_skyseg_pipeline\data\raw\285199_img_roi.jpg"
ANN = r"E:\fisheye_skyseg_pipeline\data\raw\285199_img_roi.json"

img = cv2.imread(IMG, cv2.IMREAD_COLOR)
h, w = img.shape[:2]
mask = load_polygon_mask(h, w, ANN, sky_label=1, debug_print=True)
print('uniq:', np.unique(mask).tolist(), 'fg_ratio:', float(mask.mean()))
cv2.imwrite('runs_debug/_mask_from_json.png', mask*255)
ov = img.copy()
ov[mask>0] = (0,0,255)
cv2.imwrite('runs_debug/_overlay.png', cv2.addWeighted(img, 0.6, ov, 0.4, 0))
print('saved to runs_debug/_mask_from_json.png & _overlay.png')
