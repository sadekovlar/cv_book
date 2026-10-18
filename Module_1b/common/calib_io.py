"""Чтение и запись yaml, которые сдают студенты."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_camera(path, camera_matrix, dist, image_size, rms, model, cols, rows, square_mm):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    storage.write("model", model)
    storage.write("K", camera_matrix)
    storage.write("D", dist)
    storage.write("image_width", int(image_size[0]))
    storage.write("image_height", int(image_size[1]))
    storage.write("rms", float(rms))
    storage.write("board_cols", int(cols))
    storage.write("board_rows", int(rows))
    storage.write("square_mm", float(square_mm))
    storage.release()


def load_camera(path):
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise SystemExit(f"Не удалось открыть калибровку: {path}")
    model = storage.getNode("model").string() or "pinhole"
    camera_matrix = storage.getNode("K").mat()
    dist = storage.getNode("D").mat()
    width = int(storage.getNode("image_width").real())
    height = int(storage.getNode("image_height").real())
    storage.release()
    if camera_matrix is None or dist is None:
        raise SystemExit(f"В {path} нет матрицы K или дисторсии D")
    return {
        "model": model,
        "K": camera_matrix,
        "D": dist,
        "image_size": (width, height),
    }


def load_full_calib(path):
    """Калиб формата city/leftImage.yml: K, D, r (Эйлер, рад), t, опционально sz."""
    path = Path(path)
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise SystemExit(f"Не удалось открыть калибровку: {path}")
    camera_matrix = storage.getNode("K").mat()
    dist = storage.getNode("D").mat()
    r_node = storage.getNode("r").mat()
    t_node = storage.getNode("t").mat()
    storage.release()
    if camera_matrix is None or dist is None:
        raise SystemExit(f"В {path} нет матрицы K или дисторсии D")
    if r_node is None or t_node is None:
        raise SystemExit(f"В {path} нет внешней позы r/t (как в Module_1a / data/city)")
    return {
        "model": "pinhole",
        "K": camera_matrix,
        "D": dist,
        "r_euler": np.asarray(r_node, dtype=np.float64).reshape(3),
        "t": np.asarray(t_node, dtype=np.float64).reshape(3),
        "source": str(path),
    }


def save_extrinsic(path, extrinsic):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    storage.write("preset", extrinsic["preset"])
    for key in ("height_m", "pitch_deg", "roll_deg", "yaw_deg", "lateral_m", "width_m", "near_m", "far_m"):
        storage.write(key, float(extrinsic[key]))
    storage.release()


def load_extrinsic(path):
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise SystemExit(f"Не удалось открыть внешнюю калибровку: {path}")
    extrinsic = {"preset": storage.getNode("preset").string() or "corridor"}
    for key in ("height_m", "pitch_deg", "roll_deg", "yaw_deg", "lateral_m", "width_m", "near_m", "far_m"):
        extrinsic[key] = float(storage.getNode(key).real())
    storage.release()
    return extrinsic


def label_image(image, lines):
    """Подпись кадра. OpenCV не рисует кириллицу, поэтому берётся системный шрифт."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        y = 28
        for line in lines:
            cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            y += 28
        return image

    font_path = _font()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(font_path, 22) if font_path else ImageFont.load_default()
    y = 12
    for line in lines:
        draw.text((14, y), line, font=font, fill=(0, 0, 0))
        draw.text((12, y - 1), line, font=font, fill=(255, 255, 255))
        y += 28
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _font():
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return None
