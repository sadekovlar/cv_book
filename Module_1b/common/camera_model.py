"""Проекция земли в кадр и вид сверху.

Земля: X вправо, Y вперёд, Z вверх. Начало — на земле под камерой.
Камера: x вправо, y вниз, z вперёд.
Тангаж больше нуля — оптическая ось наклонена вниз.
Крен больше нуля — правый борт вниз.
Рыскание больше нуля — поворот влево.
"""

from __future__ import annotations

import numpy as np
import cv2


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rotate_around(axis: np.ndarray, vec: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return (
        vec * np.cos(angle)
        + np.cross(axis, vec) * np.sin(angle)
        + axis * np.dot(axis, vec) * (1.0 - np.cos(angle))
    )


def ground_to_camera(pitch_deg: float, roll_deg: float, yaw_deg: float) -> np.ndarray:
    """Матрица 3x3: вектор земли -> вектор камеры."""
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)
    yaw = np.deg2rad(yaw_deg)

    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 0.0, -1.0])
    z_axis = np.array([0.0, 1.0, 0.0])

    yaw_m = _rot_z(yaw)
    x_axis, y_axis, z_axis = yaw_m @ x_axis, yaw_m @ y_axis, yaw_m @ z_axis

    # Положительный тангаж опускает оптическую ось к земле.
    y_axis = _rotate_around(x_axis, y_axis, -pitch)
    z_axis = _rotate_around(x_axis, z_axis, -pitch)

    x_axis = _rotate_around(z_axis, x_axis, roll)
    y_axis = _rotate_around(z_axis, y_axis, roll)

    cam_to_ground = np.column_stack([x_axis, y_axis, z_axis])
    return cam_to_ground.T


def pose(height_m: float, pitch_deg: float, roll_deg: float, yaw_deg: float, lateral_m: float = 0.0):
    """Поза камеры: rvec, tvec для cv2.projectPoints, матрица R и центр в земле."""
    rotation = ground_to_camera(pitch_deg, roll_deg, yaw_deg)
    center = np.array([lateral_m, 0.0, height_m], dtype=np.float64)
    translation = -rotation @ center
    rvec, _ = cv2.Rodrigues(rotation)
    return rvec, translation.reshape(3, 1), rotation, center


def project_points(points, rvec, tvec, camera_matrix, dist, model: str) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 3)
    if model == "fisheye":
        image_points, _ = cv2.fisheye.projectPoints(pts, rvec, tvec, camera_matrix, dist)
    else:
        image_points, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix, dist)
    return image_points.reshape(-1, 2)


def parallel_lines(width_m: float, near_m: float, far_m: float, step_m: float = 0.25):
    """Две линии на земле, симметричные относительно оси Y. width_m — расстояние между ними."""
    ys = np.arange(near_m, far_m + step_m * 0.5, step_m, dtype=np.float64)
    half = width_m / 2.0
    left = np.stack([np.full_like(ys, -half), ys, np.zeros_like(ys)], axis=1)
    right = np.stack([np.full_like(ys, half), ys, np.zeros_like(ys)], axis=1)
    return left, right


def ground_grid(half_width_m: float, near_m: float, far_m: float, step_m: float = 1.0):
    """Клетки на земле: продольные и поперечные отрезки."""
    lines = []
    x = -half_width_m
    while x <= half_width_m + 1e-6:
        lines.append(np.array([[x, near_m, 0.0], [x, far_m, 0.0]], dtype=np.float64))
        x += step_m
    y = near_m
    while y <= far_m + 1e-6:
        lines.append(
            np.array([[-half_width_m, y, 0.0], [half_width_m, y, 0.0]], dtype=np.float64)
        )
        y += step_m
    return lines


def _polyline(image, points_3d, rvec, tvec, camera_matrix, dist, model, color, thickness):
    if len(points_3d) < 2:
        return
    pixels = project_points(points_3d, rvec, tvec, camera_matrix, dist, model)
    cam = cv2.Rodrigues(rvec)[0] @ points_3d.T + tvec.reshape(3, 1)
    valid = cam[2] > 0.3
    h, w = image.shape[:2]
    started = False
    prev = None
    for pixel, ok in zip(pixels, valid):
        u, v = int(round(pixel[0])), int(round(pixel[1]))
        inside = ok and -w < u < 2 * w and -h < v < 2 * h
        if inside and started and prev is not None:
            cv2.line(image, prev, (u, v), color, thickness, cv2.LINE_AA)
        started = inside
        prev = (u, v) if inside else None


def draw_ground(image, camera_matrix, dist, model, extrinsic, grid=True):
    """Рисует сетку и две опорные линии. extrinsic — словарь параметров."""
    canvas = image.copy()
    rvec, tvec, _, _ = pose(
        extrinsic["height_m"],
        extrinsic["pitch_deg"],
        extrinsic["roll_deg"],
        extrinsic["yaw_deg"],
        extrinsic.get("lateral_m", 0.0),
    )
    if grid:
        reach = max(extrinsic["width_m"], 1.0)
        for segment in ground_grid(reach, extrinsic["near_m"], extrinsic["far_m"]):
            _polyline(canvas, segment, rvec, tvec, camera_matrix, dist, model, (180, 180, 80), 1)
    left, right = parallel_lines(extrinsic["width_m"], extrinsic["near_m"], extrinsic["far_m"])
    _polyline(canvas, left, rvec, tvec, camera_matrix, dist, model, (0, 180, 0), 2)
    _polyline(canvas, right, rvec, tvec, camera_matrix, dist, model, (0, 0, 220), 2)
    return canvas


def undistort(image, camera_matrix, dist, model: str):
    """Выправляет кадр и возвращает новую матрицу, которой надо пользоваться дальше."""
    h, w = image.shape[:2]
    if model == "fisheye":
        new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix, dist, (w, h), np.eye(3), balance=0.2
        )
        fixed = cv2.fisheye.undistortImage(image, camera_matrix, dist, Knew=new_k)
        return fixed, new_k
    new_k, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (w, h), 0, (w, h))
    fixed = cv2.undistort(image, camera_matrix, dist, None, new_k)
    return fixed, new_k


def birdseye(image, camera_matrix, dist, model, extrinsic, margin_m=0.6, pixels_per_meter=40):
    """Вид сверху на прямоугольник земли между near и far.

    Сначала кадр выправляется, затем строится гомография плоскости Z=0.
    """
    fixed, new_k = undistort(image, camera_matrix, dist, model)
    zero = np.zeros((4, 1), dtype=np.float64) if model == "fisheye" else np.zeros((1, 5), dtype=np.float64)
    half = extrinsic["width_m"] / 2.0 + margin_m
    x_min, x_max = -half, half
    y_min, y_max = extrinsic["near_m"], extrinsic["far_m"]
    ground = np.array(
        [
            [x_min, y_max, 0.0],
            [x_max, y_max, 0.0],
            [x_max, y_min, 0.0],
            [x_min, y_min, 0.0],
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
    # На выправленном кадре дисторсии уже нет.
    pixels = project_points(ground, rvec, tvec, new_k, zero, "pinhole")
    width = max(int(round((x_max - x_min) * pixels_per_meter)), 20)
    height = max(int(round((y_max - y_min) * pixels_per_meter)), 20)
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(pixels.astype(np.float32), destination)
    top = cv2.warpPerspective(fixed, homography, (width, height))
    return top, pixels_per_meter


def approximate_matrix(width: int, height: int) -> np.ndarray:
    """Грубая матрица, если своей калибровки камеры ещё нет. Фокус — около ширины кадра."""
    focal = 0.9 * width
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


# --- Проекция как в Module_1a (srccam.Calib / Camera): r — Эйлер в радианах, t — метры ---

_CAM_TO_VR = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def rotation_matrix_from_euler(angles_rad) -> np.ndarray:
    """Rz @ Ry @ Rx, углы в радианах [roll, pitch, yaw] — как srccam.Calib."""
    roll, pitch, yaw = [float(a) for a in np.asarray(angles_rad).reshape(3)]
    sx, cx = np.sin(roll), np.cos(roll)
    sy, cy = np.sin(pitch), np.cos(pitch)
    sz, cz = np.sin(yaw), np.cos(yaw)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def project_points_city(points_xyz, camera_matrix, r_euler, t_vec) -> np.ndarray:
    """Земля x-вправо, y-вперёд, z-вверх → пиксели. Та же формула, что Module_1a Camera."""
    rotation = rotation_matrix_from_euler(r_euler).T
    t_vec = np.asarray(t_vec, dtype=np.float64).reshape(3)
    k = np.asarray(camera_matrix, dtype=np.float64)
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    out = []
    for point in pts:
        rotated = rotation @ point - t_vec
        vr = _CAM_TO_VR @ rotated
        res = k @ vr
        w = res[2]
        if abs(w) < 1e-6:
            out.append([0.0, 0.0])
        else:
            out.append([res[0] / w, res[1] / w])
    return np.asarray(out, dtype=np.float64)


def draw_rails_city(image, camera_matrix, r_euler, t_vec, half_width_m, near_m, far_m):
    """Две линии колеи, как WayEstimator в task_1."""
    canvas = image.copy()
    left = np.array([[-half_width_m, near_m, 0.0], [-half_width_m, far_m, 0.0]])
    right = np.array([[half_width_m, near_m, 0.0], [half_width_m, far_m, 0.0]])
    for segment, color in ((left, (0, 180, 0)), (right, (0, 0, 220))):
        pix = np.round(project_points_city(segment, camera_matrix, r_euler, t_vec)).astype(np.int32)
        cv2.line(canvas, tuple(pix[0]), tuple(pix[1]), color, 3, cv2.LINE_AA)
    return canvas


def birdseye_city(image, camera_matrix, r_euler, t_vec, near_m, far_m, half_width_m, pixels_per_meter=25):
    """BEV по позе из калиба city (r, t), область задаётся метрами на земле."""
    ground = np.array(
        [
            [-half_width_m, far_m, 0.0],
            [half_width_m, far_m, 0.0],
            [half_width_m, near_m, 0.0],
            [-half_width_m, near_m, 0.0],
        ],
        dtype=np.float64,
    )
    pixels = project_points_city(ground, camera_matrix, r_euler, t_vec).astype(np.float32)
    width = max(int(round(2.0 * half_width_m * pixels_per_meter)), 20)
    height = max(int(round((far_m - near_m) * pixels_per_meter)), 20)
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(pixels, destination)
    top = cv2.warpPerspective(image, homography, (width, height))
    return top, pixels_per_meter, homography
