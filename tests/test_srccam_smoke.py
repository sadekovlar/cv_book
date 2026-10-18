"""Минимальная проверка, что пакет srccam ставится и базовые классы работают."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


def test_import_srccam():
    import srccam
    from srccam import Calib, CalibReader, Camera, Point3d, SeasonReader

    assert hasattr(srccam, "Calib")
    assert CalibReader is not None
    assert SeasonReader is not None
    assert Point3d((0.0, 1.0, 0.0)).y == 1.0


def _synthetic_calib_dict():
    k = np.array([[700.0, 0.0, 480.0], [0.0, 700.0, 270.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    d = np.zeros((5, 1), dtype=np.float64)
    # roll, pitch, yaw — радианы, как в data/city/leftImage.yml
    r = np.array([[-0.1], [0.03], [0.0]], dtype=np.float64)
    t = np.array([[0.0], [-0.5], [2.0]], dtype=np.float64)
    return {"K": k, "D": d, "r": r, "t": t}


def test_calib_and_project_point():
    from srccam import Calib, Camera, Point3d

    calib = Calib(_synthetic_calib_dict())
    camera = Camera(calib)
    u, v = camera.project_point_3d_to_2d(Point3d((0.0, 10.0, 0.0)))
    assert isinstance(u, int) and isinstance(v, int)
    # точка впереди на земле должна попасть примерно в центр кадра по x
    assert 300 < u < 660
    assert 0 < v < 540


def test_calib_reader_roundtrip(tmp_path: Path):
    from srccam import CalibReader

    yaml_path = tmp_path / "leftImage.yml"
    data = _synthetic_calib_dict()
    storage = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_WRITE)
    for key, value in data.items():
        storage.write(key, value)
    storage.release()

    reader = CalibReader(file_name=str(yaml_path), param=["K", "D", "r", "t"])
    loaded = reader.read()
    assert set(loaded) == {"K", "D", "r", "t"}
    np.testing.assert_allclose(loaded["K"], data["K"], rtol=1e-6)
    np.testing.assert_allclose(loaded["t"], data["t"], rtol=1e-6)
