import cv2
import numpy as np
import time

# Создание детектора ключевых точек
detector = cv2.ORB_create()

# Создание объекта матчинга ключевых точек
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Загрузка исходного видео
cap = cv2.VideoCapture('../data/city/trm.169.008.avi')

# Чтение первого кадра
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Поиск ключевых точек на первом кадре
prev_keypoints, prev_descriptors = detector.detectAndCompute(prev_gray, None)

while True:
    # Чтение следующего кадра
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Поиск ключевых точек на текущем кадре
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    # Матчинг ключевых точек между текущим и предыдущим кадрами
    matches = matcher.match(descriptors, prev_descriptors)

    # Вычисление вектора перемещения
    total_movement = 0
    for match in matches:
        # Получение координат ключевых точек в текущем и предыдущем кадрах
        prev_pt = prev_keypoints[match.trainIdx].pt
        curr_pt = keypoints[match.queryIdx].pt

        # Вычисление расстояния между точками
        movement = np.sqrt((curr_pt[0] - prev_pt[0])**2 + (curr_pt[1] - prev_pt[1])**2)

        # Суммирование перемещения по всем точкам
        total_movement += movement

        # Отрисовка вектора перемещения на кадре
        cv2.arrowedLine(frame, (int(prev_pt[0]), int(prev_pt[1])), (int(curr_pt[0]), int(curr_pt[1])), (0, 255, 0), 2)

    # Отрисовка общего перемещения на кадре
    cv2.putText(frame, f"Movement: {total_movement}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Вывод текущего кадра
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Обновление предыдущего кадра и ключевых точек
    prev_gray = gray
    prev_keypoints = keypoints
    prev_descriptors = descriptors

    # Задержка полторы секунды секунды
    time.sleep(0.5)

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
