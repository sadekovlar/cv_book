import itertools

import cv2
import numpy as np

from .calib import Calib
from .point import Point3d as Point


class Object3d:
    """Класс для формирования параллелепипеда заданных размеров."""

    def __init__(
        self,
        pos: Point,
        eulerRot: np.array,
        width=1.0,
        height=1.0,
        length=1.0,
        colorVertex: tuple = (255, 0, 0),
        colorEdge: tuple = (0, 255, 0),
    ):
        self._pos = pos  # Позиция объекта
        self._rotation = eulerRot  # Угол поворота объекта
        self._size = np.array(
            [width, height, length]
        )  # Размер объекта (ширина, высота, длина)
        self._colors = (colorVertex, colorEdge)  # Цвет вершин и граней/диагоналей
        self.reCalcPoints()  # Перерасчёт точек с учётом всех текущих параметров

    @property
    def pos(self):
        """Параметр: позиция параллелепипеда."""
        return self._pos

    @pos.setter
    def pos(self, value: Point):
        self._pos = value
        self.reCalcPoints()

    @property
    def rotation(self):
        """Параметр: угол поворота параллелепипеда."""
        return self._rotation

    @rotation.setter
    def rotation(self, value: np.array):
        self._rotation = value
        self.reCalcPoints()

    def add_rotation(self, value: np.array):
        """Функция: к текущему углу поворота прибавляет указанное значение."""
        self._rotation = self._rotation + value
        self.reCalcPoints()

    @property
    def size(self):
        """Параметр: размер параллелепипеда."""
        return self._size

    @size.setter
    def size(self, value: np.array):
        self._size = value
        self.reCalcPoints()

    @property
    def color(self):
        """Параметр: цвет вершин и граней."""
        return self._colors

    @color.setter
    def color(self, value: tuple):
        self._colors = value

    def reCalcPoints(self):
        """Функция для переопределения координат вершин в случае изменения параметров объекта."""
        self.points = []
        R = Calib.rotation_matrix_from(self.rotation)
        for t in itertools.product(*([[-0.5, 0.5]] * 3)):
            self.points.append(self.pos.vec.T + R.dot(t) * self.size)

    def draw(self, img, camera, drawVertex=True, drawEdges=True):
        """Функция отрисовки вершин и граней данной фигуры."""
        if drawVertex:  # Отрисовка вершин
            for i in self.points:
                cv2.circle(
                    img, camera.project_point_3d_to_2d(Point(i[0])), 3, self.color[0], 2
                )
        if drawEdges:  # Отрисовка граней
            for t in itertools.combinations(range(len(self.points)), 2):
                p1 = camera.project_point_3d_to_2d(Point(self.points[t[0]][0]))
                p2 = camera.project_point_3d_to_2d(Point(self.points[t[1]][0]))
                cv2.line(img, p1, p2, self.color[1])
