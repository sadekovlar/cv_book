#!/usr/bin/env python3
"""Подбор внешней калибровки по двум параллельным линиям на земле.

В аудитории это рельсы (ширина колеи 1520 мм). Дома те же ползунки,
но пресет corridor: кромки коридора или бордюра, ширину измеряют рулеткой.
Клавиша s сохраняет extrinsic.yaml и рисунок 02_overlay.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.calib_io import load_camera, save_extrinsic, _font
from common.camera_model import approximate_matrix, draw_ground

PRESETS = {
    "rails": {
        "preset": "rails",
        "height_m": 2.6,
        "pitch_deg": 12.0,
        "roll_deg": 0.0,
        "yaw_deg": 0.0,
        "lateral_m": 0.0,
        "width_m": 1.52,
        "near_m": 4.0,
        "far_m": 30.0,
    },
    "corridor": {
        "preset": "corridor",
        "height_m": 1.40,
        "pitch_deg": 22.0,
        "roll_deg": 0.0,
        "yaw_deg": 0.0,
        "lateral_m": 0.0,
        "width_m": 1.80,
        "near_m": 1.0,
        "far_m": 8.0,
    },
}

NAMES = {"rails": "рельсы", "corridor": "коридор / бордюр"}


def parse_args():
    parser = argparse.ArgumentParser(description="Внешняя калибровка по линиям на земле")
    parser.add_argument("--video", default="", help="Видео или картинка. Пусто — синтетический пример")
    parser.add_argument("--camera", default="", help="camera.yaml из шага с шахматкой")
    parser.add_argument("--out", default="output")
    parser.add_argument("--preset", choices=("rails", "corridor"), default="rails")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--headless", action="store_true", help="Сохранить рисунок и yaml без окна")
    parser.add_argument("--demo", action="store_true", help="Нарисовать коридор известной камерой и подобрать ту же позу")
    return parser.parse_args()


def read_frame(path: Path, index: int):
    image = cv2.imread(str(path))
    if image is not None:
        return image, None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Не открывается: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"В {path} нет кадров")
    return frame, cap


def render(frame, camera, extrinsic):
    return draw_ground(frame, camera["K"], camera["D"], camera["model"], extrinsic)


def demo_frame(extrinsic):
    """Серый пол коридора. Опорные линии при тех же параметрах ложатся на его кромки."""
    from common.camera_model import pose, project_points

    width, height = 960, 540
    camera = {
        "model": "pinhole",
        "K": approximate_matrix(width, height),
        "D": np.zeros((1, 5), dtype=np.float64),
        "image_size": (width, height),
    }
    canvas = np.full((height, width, 3), 40, np.uint8)
    half = extrinsic["width_m"] / 2.0
    near, far = extrinsic["near_m"], extrinsic["far_m"]
    quad = np.array(
        [
            [-half, near, 0.0],
            [half, near, 0.0],
            [half, far, 0.0],
            [-half, far, 0.0],
        ],
        dtype=np.float64,
    )
    rvec, tvec, _, _ = pose(
        extrinsic["height_m"],
        extrinsic["pitch_deg"],
        extrinsic["roll_deg"],
        extrinsic["yaw_deg"],
        extrinsic.get("lateral_m", 0.0),
    )
    pixels = project_points(quad, rvec, tvec, camera["K"], camera["D"], "pinhole")
    cv2.fillConvexPoly(canvas, np.round(pixels).astype(np.int32), (190, 185, 170))
    return canvas, camera


# Свои ползунки в 2 колонки: OpenCV HighGUI умеет только один столбец trackbar.
SLIDER_DEFS = [
    # key, название, формат значения, min, max
    ("height_m", "высота", "{:.2f} м", 0.30, 5.00),
    ("pitch_deg", "тангаж", "{:.1f} °", 0.0, 70.0),
    ("roll_deg", "крен", "{:.1f} °", -20.0, 20.0),
    ("yaw_deg", "рыскание", "{:.1f} °", -20.0, 20.0),
    ("lateral_m", "сдвиг", "{:.2f} м", -2.00, 2.00),
    ("width_m", "ширина", "{:.2f} м", 0.30, 5.00),
]


class SliderPanel:
    """Панель 2×3: у каждого ползунка те же подписи, что раньше на кадре."""

    COLS = 2
    ROW_H = 52
    PAD = 12
    LABEL_W = 200
    TRACK_H = 8

    def __init__(self, preset_name: str):
        start = PRESETS[preset_name]
        self.preset = preset_name
        self.values = {key: float(start[key]) for key, *_ in SLIDER_DEFS}
        self.active = None
        self.hits = []  # (key, x0, x1, y0, y1)
        self.height = self.PAD * 2 + self.ROW_H * ((len(SLIDER_DEFS) + self.COLS - 1) // self.COLS)

    def extrinsic(self):
        base = PRESETS[self.preset]
        data = {
            "preset": self.preset,
            "near_m": base["near_m"],
            "far_m": base["far_m"],
        }
        data.update(self.values)
        data["width_m"] = max(self.values["width_m"], 0.30)
        return data

    def apply_preset(self, preset_name: str):
        self.preset = preset_name
        start = PRESETS[preset_name]
        for key, *_rest in SLIDER_DEFS:
            self.values[key] = float(start[key])

    def render(self, width: int) -> np.ndarray:
        rows = (len(SLIDER_DEFS) + self.COLS - 1) // self.COLS
        panel = np.full((self.PAD * 2 + self.ROW_H * rows, width, 3), 32, np.uint8)
        self.hits = []
        col_w = width // self.COLS

        try:
            from PIL import Image, ImageDraw, ImageFont

            font_path = _font()
            font = ImageFont.truetype(font_path, 20) if font_path else ImageFont.load_default()
            pil = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)

            for index, (key, title, fmt, vmin, vmax) in enumerate(SLIDER_DEFS):
                row, col = divmod(index, self.COLS)
                x0 = col * col_w + self.PAD
                y0 = self.PAD + row * self.ROW_H
                value = self.values[key]
                label = f"{title}  {fmt.format(value)}"
                draw.text((x0, y0), label, font=font, fill=(236, 236, 236))

                track_x0 = x0 + self.LABEL_W
                track_x1 = (col + 1) * col_w - self.PAD
                track_y = y0 + 28
                if track_x1 - track_x0 < 40:
                    track_x0 = x0
                    track_y = y0 + 30
                draw.rounded_rectangle(
                    [track_x0, track_y, track_x1, track_y + self.TRACK_H],
                    radius=4,
                    fill=(70, 70, 70),
                )
                t = 0.0 if vmax == vmin else (value - vmin) / (vmax - vmin)
                t = float(np.clip(t, 0.0, 1.0))
                knob_x = track_x0 + t * (track_x1 - track_x0)
                r = 7
                draw.ellipse(
                    [knob_x - r, track_y + self.TRACK_H / 2 - r, knob_x + r, track_y + self.TRACK_H / 2 + r],
                    fill=(80, 180, 255),
                )
                self.hits.append((key, track_x0, track_x1, track_y - 10, track_y + self.TRACK_H + 10, vmin, vmax))

            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except ImportError:
            return panel

    def on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for key, x0, x1, y0, y1, vmin, vmax in self.hits:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    self.active = (key, x0, x1, vmin, vmax)
                    self._set_from_x(x)
                    return
            self.active = None
        elif event == cv2.EVENT_MOUSEMOVE and self.active is not None:
            self._set_from_x(x)
        elif event == cv2.EVENT_LBUTTONUP:
            self.active = None

    def _set_from_x(self, x):
        key, x0, x1, vmin, vmax = self.active
        t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        self.values[key] = float(vmin + np.clip(t, 0.0, 1.0) * (vmax - vmin))


def run_gui(frame, cap, camera, preset_name, out: Path):
    window = "extrinsic"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1100, 700)
    panel = SliderPanel(preset_name)
    state = {"frame": frame}

    def on_mouse(event, x, y, flags, param):
        # координаты относительно всего кадра (панель сверху)
        panel.on_mouse(event, x, y, flags, param)

    cv2.setMouseCallback(window, on_mouse)
    print("s — сохранить, n/p — кадр, 1 — рельсы, 2 — коридор, q — выход")
    print("Тяните кружки на панели (2 ряда подписей с ° и м).")
    while True:
        extrinsic = panel.extrinsic()
        view = render(state["frame"], camera, extrinsic)
        show = view
        if show.shape[1] > 1280:
            scale = 1280 / show.shape[1]
            show = cv2.resize(show, None, fx=scale, fy=scale)
        controls = panel.render(show.shape[1])
        canvas = np.vstack([controls, show])
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            save_result(out, view, extrinsic)
        if key == ord("1"):
            panel.apply_preset("rails")
        if key == ord("2"):
            panel.apply_preset("corridor")
        if cap is not None and key in (ord("n"), ord("p")):
            step = 15 if key == ord("n") else -15
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) + step
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(pos, 0))
            ok, nxt = cap.read()
            if ok:
                state["frame"] = nxt
    cv2.destroyAllWindows()
    if cap is not None:
        cap.release()


def save_result(out: Path, view, extrinsic):
    out.mkdir(parents=True, exist_ok=True)
    save_extrinsic(out / "extrinsic.yaml", extrinsic)
    cv2.imwrite(str(out / "02_overlay.png"), view)
    print(f"Сохранено {out / 'extrinsic.yaml'} и {out / '02_overlay.png'}")
    print(
        f"высота {extrinsic['height_m']:.2f} м, тангаж {extrinsic['pitch_deg']:.1f}°, "
        f"крен {extrinsic['roll_deg']:.1f}°, рыскание {extrinsic['yaw_deg']:.1f}°"
    )


def main():
    args = parse_args()
    out = Path(args.out)
    preset = PRESETS[args.preset].copy()
    cap = None
    if args.demo or not args.video:
        preset = PRESETS["corridor"].copy()
        frame, camera = demo_frame(preset)
    else:
        frame, cap = read_frame(Path(args.video), args.frame)
        if args.camera:
            camera = load_camera(args.camera)
        else:
            h, w = frame.shape[:2]
            camera = {
                "model": "pinhole",
                "K": approximate_matrix(w, h),
                "D": np.zeros((1, 5), dtype=np.float64),
                "image_size": (w, h),
            }
            print("camera.yaml не задан: взята приближённая матрица по размеру кадра.")
            print("Для отчёта сначала откалибруйте телефон по шахматке.")
    if args.headless or args.demo:
        out.mkdir(parents=True, exist_ok=True)
        if args.demo or not args.video:
            cv2.imwrite(str(out / "demo_corridor.png"), frame)
        view = render(frame, camera, preset)
        save_result(out, view, preset)
        return
    run_gui(frame, cap, camera, preset["preset"], out)


if __name__ == "__main__":
    main()
