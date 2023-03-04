import cv2
import numpy as np


class Plotter:

    @staticmethod
    def draw_x_axis(img, origin):
        x_axis = img[origin[1]]
        for pixel_number in range(len(x_axis)):
            img[origin[1]][pixel_number] = [150, 150, 150]

    @staticmethod
    def draw_y_axis(img, origin):
        for i in range(img.shape[0]):
            img[i][origin[0]] = [150, 150, 150]

    @staticmethod
    def isInFrame(value, img):
        return 0 <= value < img.shape[0]

    @staticmethod
    def draw(img,
             function,  # функция
             origin=(0, 0),  # ось координат
             scale_x=50,  # масштаб единицы по Х в пикселях
             scale_y=50,  # масштаб единицы по Y в пикселях
             color=(255, 50, 50),  # цвет прорисовки графика
             show_x_axis=True,  # отобразить ось Х
             show_y_axis=True):  # отобразить ось Y

        f = function
        origin_x = origin[0]
        origin_y = origin[1]

        if show_x_axis:
            Plotter.draw_x_axis(img, origin)

        if show_y_axis:
            Plotter.draw_y_axis(img, origin)

        x_values = np.divide(np.subtract(list(range(img.shape[1])), origin_x), scale_x)
        y_values = [f(x) for x in x_values]

        for idx in range(img.shape[1] - 1):
            if y_values == 0 or np.isnan(y_values[idx]) or np.isnan(y_values[idx + 1]):
                continue

            scaled_value = y_values[idx] * scale_y
            if np.isinf(scaled_value):
                scaled_value = img.shape[1] + 1

            scaled_value_next = y_values[idx + 1] * scale_y
            if np.isinf(scaled_value_next):
                scaled_value_next = img.shape[1] + 1

            y_pixel = origin_y - int(scaled_value)
            y_pixel_next = origin_y - int(scaled_value_next)

            if Plotter.isInFrame(y_pixel, img) and Plotter.isInFrame(y_pixel_next, img):
                cv2.line(img,
                         pt1=(idx, y_pixel),
                         pt2=(idx + 1, y_pixel_next),
                         color=color,
                         thickness=2)


def func1(x):
    return 3 * np.sin(x) / x

def func2(x):
    return 0.1 * x ** 4 - 1.5 * x ** 2 + 3

def hyperbole(x):
    return 5 / x

def sign(x):
    return 0 if x == 0 else x / abs(x)

def sqrt(x):
    return np.sqrt(5 - x)

def fence(x):
    return np.arctan(10 ** 10 * np.sin(x + np.pi / 2))


image = np.full((1000, 2000, 3), 255, dtype=np.uint8)
Plotter.draw(image, func1, (1000, 500))
# Plotter.draw(image, func2, (1000, 500), scale_x=100, scale_y=100, color=(10, 100, 10))
Plotter.draw(image, hyperbole, (1000, 500), scale_x=100, scale_y=100, color=(100, 10, 150))
Plotter.draw(image, fence, (1000, 500), scale_x=100, scale_y=100, color=(10, 200, 150))
# Plotter.draw(image, sign, (1000, 500), scale_x=100, scale_y=100, color=(10, 10, 10))
# Plotter.draw(image, sqrt, (1000, 500), scale_x=100, scale_y=100, color=(255, 200, 10))

cv2.imshow("graph", image)
cv2.waitKey()
cv2.destroyAllWindows()
