"""Сборка панорамы из изображений data/panorama с визуализацией.

Угловые детекторы OpenCV, дескриптор у всех — ORB:
    gftt   — углы Shi-Tomasi (goodFeaturesToTrack)
    harris — углы Харриса
    orb    — углы FAST

Запуск из корня репозитория:
    python Module_4a/panorama.py --detector gftt
Результат: data/panorama/out/<детектор>/
"""
import argparse
import glob
import os

import cv2
import numpy as np


# ---------------------------------------------------------------- детекторы
def _orb_describe(gray, kps):
    orb = cv2.ORB_create()
    for kp in kps:
        kp.size = 31.0
    kps, des = orb.compute(gray, kps)
    return kps, des


def detect_gftt(gray, max_kp, harris=False):
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_kp, qualityLevel=0.01,
                                  minDistance=7, blockSize=7,
                                  useHarrisDetector=harris, k=0.04)
    kps = [cv2.KeyPoint(float(x), float(y), 31) for x, y in pts.reshape(-1, 2)]
    return _orb_describe(gray, kps)


def detect_orb(gray, max_kp):
    return cv2.ORB_create(nfeatures=max_kp).detectAndCompute(gray, None)


def make_detector(args):
    if args.detector == 'orb':
        return detect_orb
    return lambda g, n: detect_gftt(g, n, harris=args.detector == 'harris')


# ---------------------------------------------------------------- сопоставление
def match(des1, des2, ratio=0.75):
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
    return [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]


def pair_homography(f1, f2, model='homography'):
    """Преобразование 3x3, переводящее точки изображения 2 в систему изображения 1.

    model='homography' — для плоских кадров, 'similarity' — для кадров на цилиндре
    (сдвиг + небольшой поворот/масштаб, устойчиво при малом числе совпадений).
    """
    good = match(f2['des'], f1['des'])
    if len(good) < 4:
        raise RuntimeError(f'Мало совпадений: {len(good)}')
    src = np.float32([f2['kps'][m.queryIdx].pt for m in good])
    dst = np.float32([f1['kps'][m.trainIdx].pt for m in good])
    if model == 'homography':
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    else:
        A, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                              ransacReprojThreshold=4.0)
        H = np.vstack([A, [0, 0, 1]])
    return H, good, mask.ravel().astype(bool)


def cylindrical_warp(img, f):
    """Проекция кадра на цилиндр радиуса f (камера вращается вокруг вертикальной оси)."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    theta = (xs - cx) / f
    hh = (ys - cy) / f
    map_x = f * np.tan(theta) + cx
    map_y = f * hh / np.cos(theta) + cy
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


# ---------------------------------------------------------------- склейка
BORDER_COLORS = [
    (0, 0, 255),     # красный
    (0, 220, 0),     # зелёный
    (255, 80, 0),    # синий
    (0, 180, 255),   # оранжевый
    (255, 0, 255),   # пурпурный
    (255, 255, 0),   # бирюзовый
]


def draw_source_borders(pano, regions, labels):
    """Контур области каждого исходного кадра на панораме, каждый своим цветом."""
    vis = pano.copy()
    for i, region in enumerate(regions):
        color = BORDER_COLORS[i % len(BORDER_COLORS)]
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, contours, -1, color, 3, cv2.LINE_AA)
    x, y = 12, 28
    for i, label in enumerate(labels):
        color = BORDER_COLORS[i % len(BORDER_COLORS)]
        cv2.rectangle(vis, (x, y - 16), (x + 22, y + 4), color, -1)
        cv2.putText(vis, label, (x + 30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (x + 30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
        y += 24
    return vis


def warp_and_blend(images, Hs, masks, labels):
    corners = []
    for img, H in zip(images, Hs):
        h, w = img.shape[:2]
        c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        corners.append(cv2.perspectiveTransform(c, H))
    allc = np.concatenate(corners)
    xmin, ymin = np.floor(allc.min(axis=(0, 1))).astype(int)
    xmax, ymax = np.ceil(allc.max(axis=(0, 1))).astype(int)
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)
    size = (int(xmax - xmin), int(ymax - ymin))
    # Гомография поворота на плоскость уводит угол за горизонт, и холст
    # раздувается до десятков тысяч пикселей. Выделение такого массива
    # выглядит как зависание.
    if size[0] > 12000 or size[1] > 12000 or size[0] * size[1] > 40_000_000:
        raise RuntimeError(
            f'Холст получился {size[0]}x{size[1]} пикселей. '
            'Плоская гомография уводит угол кадра далеко за снимок, '
            'и склейка не помещается в память. '
            'Для этого поворота запустите --projection cylindrical.'
        )
    print(f'Холст {size[0]}x{size[1]}', flush=True)

    acc = np.zeros((size[1], size[0], 3), np.float32)
    wsum = np.zeros((size[1], size[0]), np.float32)
    for img, H, m in zip(images, Hs, masks):
        # вес = расстояние до края кадра -> плавный переход (feathering)
        m = m.copy()
        m[[0, -1], :] = 0
        m[:, [0, -1]] = 0
        weight = cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float32)
        warped = cv2.warpPerspective(img, T @ H, size).astype(np.float32)
        wwarp = cv2.warpPerspective(weight, T @ H, size)
        acc += warped * wwarp[..., None]
        wsum += wwarp
    pano = (acc / np.maximum(wsum, 1e-6)[..., None]).astype(np.uint8)

    regions = []
    for m, H in zip(masks, Hs):
        region = cv2.warpPerspective((m > 0).astype(np.uint8) * 255, T @ H, size,
                                     flags=cv2.INTER_NEAREST)
        regions.append(region)

    ys, xs = np.nonzero(wsum > 0)  # обрезка пустых полей по ограничивающему прямоугольнику
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pano = pano[y0:y1, x0:x1]
    regions = [r[y0:y1, x0:x1] for r in regions]
    return pano, draw_source_borders(pano, regions, labels), (int(x0), int(y0), int(xmin), int(ymin))


def cylinder_to_perspective(pano, f, cx, cy, max_angle_deg=75):
    """Развернуть цилиндрическую панораму на плоскость.

    Оптическая ось смотрит в середину дуги, а не в опорный кадр: иначе
    дальний край оказывается у 90° и одна стена занимает весь снимок.
    На плоскости x = f·tan(β), y = cy + (y_цил − cy) / cos(β),
    где β — угол от середины дуги. Прямые комнаты снова становятся прямыми.
    """
    valid = np.any(pano > 0, axis=2)
    ys, xs = np.nonzero(valid)
    theta = (xs.astype(np.float64) - cx) / f
    alpha = 0.5 * (theta.min() + theta.max())
    beta = theta - alpha
    keep = np.abs(beta) < np.deg2rad(max_angle_deg)
    beta, ys = beta[keep], ys[keep]
    if beta.size == 0:
        raise RuntimeError('На цилиндре нет точек внутри допустимого угла обзора.')
    u = f * np.tan(beta)
    v = (ys - cy) / np.cos(beta)
    umin, umax = np.floor(u.min()), np.ceil(u.max())
    vmin, vmax = np.floor(v.min()), np.ceil(v.max())
    out_w, out_h = int(umax - umin), int(vmax - vmin)
    if out_w > 12000 or out_h > 12000 or out_w * out_h > 40_000_000:
        raise RuntimeError(
            f'Плоская развёртка получилась {out_w}x{out_h}. '
            'Край панорамы слишком близко к 90° от середины дуги.'
        )
    half = np.degrees(np.max(np.abs(beta)))
    print(f'Плоская развёртка {out_w}x{out_h}, край ±{half:.0f}°', flush=True)
    uu, vv = np.meshgrid(
        np.arange(out_w, dtype=np.float32) + np.float32(umin),
        np.arange(out_h, dtype=np.float32) + np.float32(vmin),
    )
    b = np.arctan2(uu, np.float32(f))
    map_x = (np.float32(cx) + np.float32(f) * (b + np.float32(alpha))).astype(np.float32)
    map_y = (np.float32(cy) + vv * np.cos(b)).astype(np.float32)
    plane = cv2.remap(pano, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = np.any(plane > 0, axis=2)
    rys, rxs = np.nonzero(mask)
    return plane[rys.min():rys.max() + 1, rxs.min():rxs.max() + 1]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/panorama')
    ap.add_argument('--out', default=None,
                    help='куда писать результат; по умолчанию <input>/out')
    ap.add_argument('--detector', default='gftt', choices=['gftt', 'harris', 'orb'])
    ap.add_argument('--max-kp', type=int, default=3000)
    ap.add_argument('--scale', type=float, default=0.5, help='масштаб входных кадров')
    ap.add_argument('--projection', default='cylindrical', choices=['cylindrical', 'plane'])
    ap.add_argument('--focal', type=float, default=0.8,
                    help='фокусное расстояние в долях ширины кадра (для телефона ~0.75-0.85)')
    ap.add_argument('--show', action='store_true', help='показывать окна cv2.imshow')
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input, '*.jpg')))
    assert len(paths) >= 2, f'Нет изображений в {args.input}'
    images = [cv2.resize(cv2.imread(p), None, fx=args.scale, fy=args.scale,
                         interpolation=cv2.INTER_AREA) for p in paths]
    masks = [np.full(img.shape[:2], 255, np.uint8) for img in images]
    if args.projection == 'cylindrical':
        f = args.focal * images[0].shape[1]
        images = [cylindrical_warp(img, f) for img in images]
        masks = [cylindrical_warp(m, f) for m in masks]
    model = 'similarity' if args.projection == 'cylindrical' else 'homography'
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    out_dir = os.path.join(args.out or os.path.join(args.input, 'out'), args.detector)
    os.makedirs(out_dir, exist_ok=True)

    def save(name, img):
        cv2.imwrite(os.path.join(out_dir, name), img)
        if args.show:
            cv2.imshow(name, img)
            cv2.waitKey(0)

    detector = make_detector(args)

    # 1. ключевые точки
    feats = []
    for i, img in enumerate(images):
        gray = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        kps, des = detector(gray, args.max_kp)
        # убираем точки на границе цилиндрической маски
        inside = cv2.erode(masks[i], np.ones((15, 15), np.uint8))
        keep = [j for j, k in enumerate(kps) if inside[int(k.pt[1]), int(k.pt[0])]]
        kps, des = [kps[j] for j in keep], des[keep]
        feats.append({'kps': kps, 'des': des})
        print(f'[{i}] {os.path.basename(paths[i])}: {len(kps)} точек')
        vis = cv2.drawKeypoints(img, kps, None, (0, 255, 0),
                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        save(f'kp_{i}.jpg', vis)

    # 2. попарные гомографии соседних кадров + визуализация инлайеров
    pair_H = []
    for i in range(len(images) - 1):
        H, good, inl = pair_homography(feats[i], feats[i + 1], model)
        pair_H.append(H)
        print(f'пара {i}-{i + 1}: совпадений {len(good)}, инлайеров {inl.sum()}')
        inliers = [m for m, ok in zip(good, inl) if ok]
        vis = cv2.drawMatches(images[i + 1], feats[i + 1]['kps'], images[i], feats[i]['kps'],
                              inliers, None, matchColor=(0, 255, 0),
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        save(f'matches_{i}_{i + 1}.jpg', vis)

    # 3. цепочка гомографий относительно центрального кадра (меньше искажений)
    ref = len(images) // 2
    Hs = [np.eye(3) for _ in images]
    for i in range(ref - 1, -1, -1):              # левее опорного: H(i) = H(i+1) @ inv(H_i,i+1)
        Hs[i] = Hs[i + 1] @ np.linalg.inv(pair_H[i])
    for i in range(ref + 1, len(images)):         # правее опорного
        Hs[i] = Hs[i - 1] @ pair_H[i - 1]

    # 4. склейка
    labels = [f'{i}  {os.path.basename(p)}' for i, p in enumerate(paths)]
    pano, borders, (x0, y0, xmin, ymin) = warp_and_blend(images, Hs, masks, labels)
    save('panorama.jpg', pano)
    save('panorama_borders.jpg', borders)
    print(f'Панорама {pano.shape[1]}x{pano.shape[0]} -> {out_dir}/panorama.jpg')
    print(f'Границы кадров -> {out_dir}/panorama_borders.jpg')
    if args.projection == 'cylindrical':
        rh, rw = images[ref].shape[:2]
        cx = rw / 2 - xmin - x0
        cy = rh / 2 - ymin - y0
        plane = cylinder_to_perspective(pano, f, cx, cy)
        save('panorama_plane.jpg', plane)
        print(f'Прямые линии -> {out_dir}/panorama_plane.jpg')
    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
