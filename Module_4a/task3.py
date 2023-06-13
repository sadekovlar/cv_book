import cv2
import numpy as np

# Создание детектора ключевых точек
detector = cv2.ORB_create()

# Загрузка исходного видео
cap = cv2.VideoCapture('output.mp4')

# Получение первого кадра
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
    
    # Определение точек на текущем кадре
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    
   # Матчинг ключевых точек между текущим и предыдущим кадрами
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(prev_descriptors, descriptors)
    
    # Вычисление вектора перемещения
    total_movement = 0
    for match in matches:
        prev_pt = prev_keypoints[match.trainIdx].pt
        curr_pt = keypoints[match.queryIdx].pt
        movement = np.sqrt((curr_pt[0] - prev_pt[0])**2 + (curr_pt[1] - prev_pt[1])**2)
        total_movement += movement

    # Отрисовка общего перемещения на кадре
    cv2.putText(frame, f"Movement: {total_movement}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Определение движения объекта
    avg_movement = total_movement / len(matches)
    if avg_movement > 4:  # Пороговое значение для определения движения
        # Отображение прямоугольников вокруг движущихся объектов
        for match in matches:
            prev_point = prev_keypoints[match.queryIdx].pt
            curr_point = keypoints[match.trainIdx].pt
            x, y = curr_point
            cv2.rectangle(frame, (int(x) - 10, int(y) - 10), (int(x) + 10, int(y) + 10), (0, 255, 0), 2)
    
    # Вывод текущего кадра
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Обновление предыдущего кадра и ключевых точек
    prev_gray = gray
    prev_keypoints = keypoints
    prev_descriptors = descriptors

cap.release()
cv2.destroyAllWindows()
