import cv2
import numpy as np
import math
import time
import random

SHOW_RES = True


# Получить последнюю точку в списке списочков и одиночных точек
def get_last_point(ls):
    lls = len(ls)
    if lls > 0:
        item = ls[lls - 1]
        if type(item) == list:
            if len(item) > 0:
                x, y = item[len(item) - 1]
            else:
                return 0, 0, False
        else:
            x, y = item
        return x, y, True
    return 0, 0, False


# Вычислить расстояние между точками
def get_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    l = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))  # Евклидово расстояние между точками
    return l


# Добавить точку в кластер
def add_point_to_claster(x, y, ls):
    lls = len(ls)
    item = ls[lls - 1]
    if type(item) == list:
        item.append((x, y))
    else:
        x1, y1 = item
        item = [(x1, y1)]
        item.append((x, y))
        ls[lls - 1] = item


def calk_center(ls):
    ix = 0
    iy = 0
    l = float(len(ls))
    for point in ls:
        x, y = point
        ix += x
        iy += y
    return round(ix / l), round(iy / l)


# получить центр масс точек
def get_center(centers, point):
    l = len(centers)
    res = -1
    min_r = 9999999999999.0
    for i in range(0, l):
        center = centers[i]
        x, y, count = center
        r = get_distance(point, (x, y))
        if r >= 10:
            continue
        if r < min_r:
            res = i
            min_r = r
    return res


def add_to_center(center, point):
    x1, y1, count = center
    count += 1
    x2, y2 = point
    x = x1 + (x2 - x1) / float(count)
    y = y1 + (y2 - y1) / float(count)
    return x, y, count


if __name__ == "__main__":
    cap = cv2.VideoCapture(r'../data/tram/trm.169.007.avi')
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        operatedImage = np.float32(gray)
        dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)
        dest = cv2.dilate(dest, None)
        img_blank = np.zeros(frame.shape)
        img_blank[dest > 0.05 * dest.max()] = [0, 0, 255]
        heigh = img_blank.shape[0]
        width = img_blank.shape[1]
        points = []
        for x in range(0, width):
            for y in range(0, heigh):
                if img_blank[y, x, 2] == 255:
                    points.append((x, y))
        # Теперь будем обрабатывать этот список
        points_count = len(points)
        print("Количество обрабатываемых точек: ", points_count)
        beg_time = time.perf_counter()
        centers = []
        for i in range(0, points_count):
            point = points[i]
            center_index = get_center(centers, point)
            if center_index == -1:
                x, y = point
                centers.append((x, y, 1))
            else:
                center = centers[center_index]
                centers[center_index] = add_to_center(center, point)
        end_time = time.perf_counter()
        print("Прошло времени ", end_time - beg_time)

        print("Осталось точек ", len(centers))

        img_blank1 = np.zeros(frame.shape)
        random_points = centers if len(centers) < 200 else random.sample(centers, 200)
        for center in random_points:
            x, y, count = center
            # frame[int(y), int(x), 2] = 255
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), 5)

        # окно с выводимым изображением
        cv2.imshow('Harris', frame)
        """
        frame[dest > 0.05 * dest.max()] = [0, 0, 255]
        img_blank = np.zeros(frame.shape)
        img_blank[dest > 0.05 * dest.max()] = [0, 0, 255]
        height = img_blank.shape[0]
        width = img_blank.shape[1]
        points = []
        for x in range(0, width):
            for y in range(0, height):
                if img_blank[y, x, 2] == 255:
                    x1, y1, point_is = get_last_point(points)
                    if point_is:
                        l = get_distance((x1, y1), (x, y))
                        if l < 3:
                            add_point_to_claster(x, y, points)
                            continue
                    points.append((x, y))
        centers = []
        for point in points:
            if type(point) == list:
                centers.append(calk_center(point))
            else:
                centers.append(point)
        img_blank1 = np.zeros(frame.shape)
        for point in centers:
            print(point)
            x, y = point
            img_blank1[y, x, 2] = 255
        cv2.imshow('frame', img_blank1)
        """
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()