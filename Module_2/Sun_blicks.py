import cv2
import numpy as np

# Функция для удаления солнечных бликов с помощью фильтров
def remove_glare(frame):
    # Преобразуем изображение в формат HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Разделение каналов
    h, s, v = cv2.split(hsv)

    # Применяем адаптивную яркость
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15, 15))
    v_clahe = clahe.apply(v)

    # Объединяем каналы обратно в изображение
    hsv_processed = cv2.merge((h, s, v_clahe))

    # Конвертируем обратно в формат BGR
    image_processed = cv2.cvtColor(hsv_processed, cv2.COLOR_HSV2BGR)

    return image_processed




# Открытие видеофайла
video_path = "../data/processing/trm.168.090.avi"
cap = cv2.VideoCapture(video_path)

# Обработка каждого кадра видео
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Вызов функции удаления солнечных бликов
    frame_without_glare = remove_glare(frame)

    # Отображение обработанного кадра
    cv2.imshow("Processed Video", frame_without_glare)

    # Прерывание обработки видео при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
