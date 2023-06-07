import cv2
import numpy as np

# Загрузка видео из папки data/tram
cap = cv2.VideoCapture('trm.169.008.avi')

while True:
    # Чтение кадра из видео
    ret, frame = cap.read()

    if not ret:
        break

    # Преобразование кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Применение фильтра Гаусса для сглаживания шума
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Улучшение контраста с помощью адаптивной гистограммной эквализации
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_equalized = clahe.apply(gray_blurred)

    # Применение цветового фильтра для выделения зеленого цвета
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Применение преобразования Хафа для окружностей
    circles = cv2.HoughCircles(
        mask,
        cv2.HOUGH_GRADIENT,
        dp=15,
        minDist=20,
        param1=6,
        param2=6,
        minRadius=1,
        maxRadius=30
    )

    # Если обнаружены окружности
    if circles is not None:
        print(len(circles))
        # Округление координат и радиусов
        circles = np.round(circles[0, :]).astype(int)

        # Отрисовка обнаруженных окружностей
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)  # Зеленый цвет для окружностей
            cv2.rectangle(frame, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)  # Отметка центра окружности

    # Отображение кадра с обнаруженными окружностями
    cv2.imshow('Traffic Lights Detection', frame)

    # Завершение работы по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
