#!/usr/bin/env python3
"""Динамический bird's-eye view по калибровке city (K, D, r, t).

Поза камеры читается из leftImage.yml — та же, что в Module_1a/task_1.
Область BEV (near/far/ширина, масштаб) задаётся в bev_config.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_1B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_1B))

from common.calib_io import label_image, load_full_calib
from common.camera_model import birdseye_city, draw_rails_city


DEFAULT_CONFIG = Path(__file__).resolve().parent / "bev_config.json"


def parse_args():
    parser = argparse.ArgumentParser(description="BEV по видео и калибу city")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="JSON с размерами области BEV")
    parser.add_argument("--calib", default="", help="Переопределить путь к leftImage.yml")
    parser.add_argument("--video", default="", help="Переопределить видео")
    parser.add_argument("--frame", type=int, default=0, help="Стартовый кадр")
    parser.add_argument("--save-frame", default="", help="Если задано — сохранить один кадр и выйти")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    required = ("near_m", "far_m", "half_width_m", "pixels_per_meter")
    for key in required:
        if key not in cfg:
            raise SystemExit(f"В {path} нет поля {key}")
    return cfg


def fit_height(image, height):
    if image.shape[0] == height:
        return image
    scale = height / max(image.shape[0], 1)
    return cv2.resize(image, (max(int(image.shape[1] * scale), 1), height))


def fit_width(image, width):
    if image.shape[1] == width:
        return image
    scale = width / max(image.shape[1], 1)
    return cv2.resize(image, (width, max(int(image.shape[0] * scale), 1)))


def compose(frame, top, cfg, calib):
    rail_half = float(cfg.get("rail_half_width_m", cfg["half_width_m"]))
    ground_half = float(cfg["half_width_m"])
    overlay = draw_rails_city(
        frame,
        calib["K"],
        calib["r_euler"],
        calib["t"],
        rail_half,
        cfg["near_m"],
        cfg["far_m"],
    )
    left = label_image(overlay, ["кадр + линии из калиба (r, t)"])

    target_w = int(cfg.get("bev_width_px") or max(int(left.shape[1] * 0.5), 320))
    right = fit_width(top, target_w)
    # если после масштаба BEV выше кадра — поджать по высоте кадра
    if right.shape[0] > left.shape[0]:
        right = fit_height(right, left.shape[0])

    right = label_image(
        right,
        [
            "вид сверху",
            f"{cfg['pixels_per_meter']} пикс/м",
            f"Y {cfg['near_m']:.0f}…{cfg['far_m']:.0f} м",
            f"полоса {2 * ground_half:.1f} м",
        ],
    )
    if cfg.get("layout", "side_by_side") == "pip":
        scale = float(cfg.get("overlay_scale", 0.45))
        small = cv2.resize(right, None, fx=scale, fy=scale)
        out = left.copy()
        y0, x0 = 8, 8
        y1, x1 = y0 + small.shape[0], x0 + small.shape[1]
        if y1 <= out.shape[0] and x1 <= out.shape[1]:
            out[y0:y1, x0:x1] = small
        return out
    left = fit_height(left, right.shape[0])
    return np.concatenate([left, right], axis=1)


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    calib_path = Path(args.calib or cfg.get("calib", "../data/city/leftImage.yml"))
    video_path = Path(args.video or cfg.get("video", "../data/city/trm.169.007.avi"))
    if not calib_path.is_absolute():
        calib_path = (cfg_path.parent / calib_path).resolve()
    if not video_path.is_absolute():
        video_path = (cfg_path.parent / video_path).resolve()

    calib = load_full_calib(calib_path)
    print(f"калиб: {calib_path}")
    print(f"r (рад) = {calib['r_euler']}")
    print(f"t (м)   = {calib['t']}")
    print(
        f"BEV: near={cfg['near_m']} far={cfg['far_m']} "
        f"half_w={cfg['half_width_m']} ppm={cfg['pixels_per_meter']}"
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Не открывается видео: {video_path}")
    if args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)

    delay = int(cfg.get("delay_ms", 30))
    paused = False
    frame = None
    window = "bev"
    print("пробел — пауза, n — кадр вперёд (на паузе), q — выход")

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)

    while True:
        if not paused or frame is None:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    break
        top, ppm, _H = birdseye_city(
            frame,
            calib["K"],
            calib["r_euler"],
            calib["t"],
            cfg["near_m"],
            cfg["far_m"],
            cfg["half_width_m"],
            pixels_per_meter=cfg["pixels_per_meter"],
        )
        pair = compose(frame, top, cfg, calib)
        if args.save_frame:
            dest = Path(args.save_frame)
            cv2.imwrite(str(dest), pair)
            print(f"кадр: {dest}")
            break
        cv2.imshow(window, pair)
        key = cv2.waitKey(0 if paused else delay) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        if paused and key == ord("n"):
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()

    # последний кадр для отчёта
    if not args.save_frame and frame is not None:
        top, _, _ = birdseye_city(
            frame,
            calib["K"],
            calib["r_euler"],
            calib["t"],
            cfg["near_m"],
            cfg["far_m"],
            cfg["half_width_m"],
            pixels_per_meter=cfg["pixels_per_meter"],
        )
        cv2.imwrite(str(out_dir / "03_bev.png"), compose(frame, top, cfg, calib))
        print(f"сохранено {out_dir / '03_bev.png'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
