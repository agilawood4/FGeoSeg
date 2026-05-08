import os, json, cv2, numpy as np

LIST = r'./data/train_list.txt'   # 按需改
N = 10                            # 抽查前 N 条
POSITIVE_LABELS = ('sky','white_region','天空','Sky','SKY')

def count_polys_from_json(data):
    """统计 JSON 中属于天空标签的多边形数量（兼容多种结构）"""
    cnt = 0

    # 1) Labelme: shapes
    shapes = data.get('shapes')
    if isinstance(shapes, list):
        for shp in shapes:
            lbl = str(shp.get('label', '')).strip()
            if lbl.lower() in [s.lower() for s in POSITIVE_LABELS]:
                pts = shp.get('points')
                if isinstance(pts, list) and len(pts) >= 3:
                    cnt += 1

    # 2) 自定义: sky
    sky = data.get('sky')
    if isinstance(sky, list):
        for poly in sky:
            if isinstance(poly, list) and len(poly) >= 3:
                cnt += 1

    # 3) 自定义: objects[{label, polygon/points}]
    objs = data.get('objects')
    if isinstance(objs, list):
        for o in objs:
            lbl = str(o.get('label', '')).strip()
            if lbl.lower() in [s.lower() for s in POSITIVE_LABELS]:
                poly = o.get('polygon') or o.get('points')
                if isinstance(poly, list) and len(poly) >= 3:
                    cnt += 1

    # 4) 自定义: polygons[{label, points/polygon}]
    polys = data.get('polygons')
    if isinstance(polys, list):
        for o in polys:
            lbl = str(o.get('label', '')).strip()
            if lbl.lower() in [s.lower() for s in POSITIVE_LABELS]:
                poly = o.get('points') or o.get('polygon')
                if isinstance(poly, list) and len(poly) >= 3:
                    cnt += 1
    return cnt

def peek_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return False, f'JSON 打不开: {e}', None
    cnt = count_polys_from_json(data)
    return True, f'找到候选多边形数: {cnt}', data

def main():
    from data.fisheye_json_dataset import load_polygon_mask  # 用项目里的函数做最终裁决

    with open(LIST, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for i, ln in enumerate(lines[:N]):
        parts = ln.split('\t')
        img_p = parts[0]
        ann_p = parts[1] if len(parts) > 1 else None

        print(f'\n[{i}] img={os.path.basename(img_p)}')
        if not ann_p:
            print('  ⚠️ 这条没有第二列（标注路径），训练时会全 0')
            continue

        exists = os.path.isfile(ann_p)
        ext = os.path.splitext(ann_p)[1].lower()
        print(f'    anno={ann_p}  exists={exists}  ext={ext}')
        if not exists:
            print('  ❌ 标注文件不存在')
            continue

        if ext in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff'):
            # 掩码图片：>0 为天空
            m = cv2.imread(ann_p, cv2.IMREAD_GRAYSCALE)
            if m is None:
                print('  ❌ 掩码图片读失败')
            else:
                u = np.unique(m)
                print(f'  ✅ 作为掩码 PNG 读取，唯一值: {u.tolist()}, fg_ratio={(m>0).mean():.4f}')
                # 统一再用 load_polygon_mask 跑一遍（与数据管线一致）
                img = cv2.imread(img_p, cv2.IMREAD_COLOR)
                if img is not None:
                    H,W = img.shape[:2]
                    # 直接把 ann_p 传给 load_polygon_mask（若其只支持 json，可在数据集里实现 PNG fallback）
                    mask = (m>0).astype(np.uint8)
                    if mask.shape[:2] != (H,W):
                        mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)
                    print(f'  >>> 与图同尺寸后 uniq={np.unique(mask).tolist()}, fg_ratio={(mask>0).mean():.4f}')
        elif ext == '.json':
            ok, msg, data = peek_json(ann_p)
            print('  JSON检查:', msg)
            # 用项目里的 load_polygon_mask 在“图像真实尺寸”上栅格化（最可靠）
            img = cv2.imread(img_p, cv2.IMREAD_COLOR)
            if img is None:
                print('  ❌ 原图读取失败，无法试栅格化')
            else:
                H,W = img.shape[:2]
                mask = load_polygon_mask(H, W, ann_p, sky_label=1, debug_print=True)
                print(f'  >>> 栅格化(原图尺寸) uniq={np.unique(mask).tolist()}, fg_ratio={(mask>0).mean():.4f}')
        else:
            print('  ⚠️ 既不是 json 也不是常见图像扩展名？')

if __name__ == '__main__':
    main()
