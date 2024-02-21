import cv2
import numpy as np

# Загрузка видео
cap = cv2.VideoCapture('trm.169.008.avi')

# Получение информации о видео
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Создание объекта для записи видео
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

# Определение параметров для алгоритма Фарнебака
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Чтение первого кадра
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Цикл обработки кадров
while True:
    # Чтение текущего кадра
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование в градации серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Вычисление оптического потока с помощью алгоритма Фарнебака
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Извлечение компонент X и Y скорости движения
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]

    # Вычисление средней скорости движения
    mean_flow_x = np.mean(flow_x)
    mean_flow_y = np.mean(flow_y)

    # Определение характера движения
    if abs(mean_flow_x) > abs(mean_flow_y):
        if mean_flow_x > 0:
            movement_direction = "Moving Left"
        else:
            movement_direction = "Moving Right"
    else:
        if abs(mean_flow_y) < 0.5:
            movement_direction = "Don't Moving"
        elif mean_flow_y > 0:
            movement_direction = "Moving Forward"
        else:
            movement_direction = "Moving Back"

    # Визуализация движения на кадре
    arrow_start = (frame.shape[1] // 2, frame.shape[0] // 2)
    arrow_end = (frame.shape[1] // 2 + int(100 * mean_flow_x), frame.shape[0] // 2 + int(100 * mean_flow_y))
    draw_frame = cv2.arrowedLine(frame, arrow_end, arrow_start, (0, 255, 0), 5)

    # Отображение характера движения
    cv2.putText(draw_frame, movement_direction, 
                ((frame.shape[1]-cv2.getTextSize(movement_direction, cv2.FONT_HERSHEY_SIMPLEX, 1, 4)[0][0]) // 2, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

    # Запись кадра в выходной видеофайл
    out.write(draw_frame)

    # Отображение кадра
    cv2.imshow("Optical Flow", draw_frame)
    if cv2.waitKey(1) == 27:  # Нажмите ESC для выхода
        break

    # Обновление предыдущего кадра и градаций серого
    prev_gray = gray

# Закрытие видеофайла и окон
cap.release()
cv2.destroyAllWindows()