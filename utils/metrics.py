# utils/metrics.py
import numpy as np

def bin_stats(pred: np.ndarray, gt: np.ndarray):
    """pred/gt: uint8或bool，1表示正类(天空)。返回TP,FP,FN,TN"""
    p = (pred > 0).astype(np.uint8)
    g = (gt   > 0).astype(np.uint8)
    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    tn = int(((p == 0) & (g == 0)).sum())
    return tp, fp, fn, tn

def safe_div(a, b, eps=1e-7):
    return a / (b + eps)

def bin_metrics(tp, fp, fn, tn):
    iou  = safe_div(tp, tp + fp + fn)
    dice = safe_div(2*tp, 2*tp + fp + fn)
    prec = safe_div(tp, tp + fp)
    rec  = safe_div(tp, tp + fn)
    acc  = safe_div(tp + tn, tp + tn + fp + fn)
    f1   = safe_div(2*prec*rec, prec + rec)
    return dict(mIoU=iou, Dice=dice, Precision=prec, Recall=rec, F1=f1, Acc=acc)

def reduce_metrics(metric_list):
    keys = metric_list[0].keys()
    return {k: float(np.mean([m[k] for m in metric_list])) for k in keys}
