#!/usr/bin/env python3
"""
Скрипт для поиска объекта в видео с использованием детектора ключевых точек SIFT.
Сравнивает каждый кадр видео с шаблоном изображения.
"""

import cv2
import numpy as np
import os
from pathlib import Path


def brighten_frame(frame, brightness_factor=1.5):
    """
    Увеличивает яркость кадра.
    
    Args:
        frame: Входной кадр (BGR)
        brightness_factor: Коэффициент яркости (1.0 = без изменений, >1.0 = ярче)
    
    Returns:
        Более светлый кадр
    """
    # Преобразуем в HSV для работы с яркостью
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Увеличиваем яркость (канал V)
    v = cv2.multiply(v, brightness_factor)
    v = np.clip(v, 0, 255).astype(np.uint8)
    
    # Объединяем обратно
    hsv_bright = cv2.merge([h, s, v])
    bright_frame = cv2.cvtColor(hsv_bright, cv2.COLOR_HSV2BGR)
    
    return bright_frame


def overlay_template(frame, template, position='bottom_right', scale=0.2, margin=10):
    """
    Накладывает шаблон на кадр в указанном углу.
    
    Args:
        frame: Исходный кадр (BGR)
        template: Шаблон (BGR или Grayscale)
        position: Позиция ('bottom_right', 'bottom_left', 'top_right', 'top_left')
        scale: Масштаб шаблона относительно кадра (0-1)
        margin: Отступ от края в пикселях
    
    Returns:
        Кадр с наложенным шаблоном
    """
    h_frame, w_frame = frame.shape[:2]
    h_template, w_template = template.shape[:2]
    
    # Вычисление размеров шаблона после масштабирования
    new_w = int(w_frame * scale)
    new_h = int(h_template * (new_w / w_template))
    
    # Масштабирование шаблона
    if len(template.shape) == 2:  # Grayscale
        template_resized = cv2.resize(template, (new_w, new_h))
        template_resized = cv2.cvtColor(template_resized, cv2.COLOR_GRAY2BGR)
    else:  # BGR
        template_resized = cv2.resize(template, (new_w, new_h))
    
    # Определение позиции
    if position == 'bottom_right':
        x = w_frame - new_w - margin
        y = h_frame - new_h - margin
    elif position == 'bottom_left':
        x = margin
        y = h_frame - new_h - margin
    elif position == 'top_right':
        x = w_frame - new_w - margin
        y = margin
    elif position == 'top_left':
        x = margin
        y = margin
    else:
        x = w_frame - new_w - margin
        y = h_frame - new_h - margin
    
    # Создание копии кадра
    result = frame.copy()
    
    # Наложение шаблона с рамкой
    # Рисуем белую рамку
    cv2.rectangle(result, (x-2, y-2), (x+new_w+2, y+new_h+2), (255, 255, 255), 2)
    # Накладываем шаблон
    result[y:y+new_h, x:x+new_w] = template_resized
    
    # Добавляем подпись
    cv2.putText(result, "Template", (x, y-5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result, (x, y, new_w, new_h)  # Возвращаем также координаты шаблона


def draw_keypoint_connections(frame, template_pos, template_original_size, 
                               kp_template, kp_frame, good_matches, 
                               template_scale=0.15, frame_scale=1.0):
    """
    Рисует связи между ключевыми точками шаблона (в углу) и соответствующими точками в кадре.
    
    Args:
        frame: Кадр для рисования (BGR)
        template_pos: Позиция шаблона (x, y, w, h)
        template_original_size: Оригинальный размер шаблона (h, w)
        kp_template: Ключевые точки шаблона
        kp_frame: Ключевые точки кадра
        good_matches: Список совпадений
        template_scale: Масштаб шаблона на кадре
        frame_scale: Масштаб кадра (если кадр был изменен)
    
    Returns:
        Кадр с нарисованными связями
    """
    if len(good_matches) == 0 or len(kp_frame) == 0 or len(kp_template) == 0:
        return frame
    
    result = frame.copy()
    x_template, y_template, w_template, h_template = template_pos
    h_orig, w_orig = template_original_size
    
    # Масштабные коэффициенты
    scale_x = w_template / w_orig
    scale_y = h_template / h_orig
    
    # Рисуем связи между точками
    max_connections = min(30, len(good_matches))  # Ограничиваем количество для читаемости
    for i, match in enumerate(good_matches[:max_connections]):
        try:
            # Координаты точки в шаблоне (в координатах оригинального шаблона)
            pt_template = kp_template[match.queryIdx].pt
            
            # Преобразуем в координаты шаблона на кадре
            template_x = int(x_template + pt_template[0] * scale_x)
            template_y = int(y_template + pt_template[1] * scale_y)
            
            # Координаты соответствующей точки в кадре
            pt_frame = kp_frame[match.trainIdx].pt
            frame_x = int(pt_frame[0] * frame_scale)
            frame_y = int(pt_frame[1] * frame_scale)
            
            # Проверка валидности координат
            h_frame, w_frame = result.shape[:2]
            if (0 <= template_x < w_frame and 0 <= template_y < h_frame and
                0 <= frame_x < w_frame and 0 <= frame_y < h_frame):
                
                # Рисуем линию между точками (более заметную)
                # Цвет зависит от качества совпадения (чем лучше совпадение, тем зеленее)
                match_quality = 1.0 - (match.distance / 300.0)  # Нормализация расстояния
                match_quality = max(0.0, min(1.0, match_quality))
                color = (0, int(255 * match_quality), int(255 * (1 - match_quality)))
                cv2.line(result, (template_x, template_y), (frame_x, frame_y), color, 2, cv2.LINE_AA)
                
                # Рисуем маленькие кружки на точках
                cv2.circle(result, (template_x, template_y), 4, (0, 255, 0), -1)  # Зеленый для шаблона
                cv2.circle(result, (frame_x, frame_y), 4, (255, 0, 0), -1)  # Синий для кадра
        except (IndexError, AttributeError):
            # Пропускаем некорректные совпадения
            continue
    
    return result


def detect_object_sift(video_path, template_path, output_path=None, match_threshold=0.6, show_preview=True, brightness_factor=1.5):
    """
    Находит объект в видео, сравнивая кадры с шаблоном используя SIFT.
    
    Args:
        video_path: Путь к видео файлу
        template_path: Путь к шаблону изображения
        output_path: Путь для сохранения результата (опционально)
        match_threshold: Порог для фильтрации совпадений (0-1)
        show_preview: Показывать ли визуализацию в реальном времени
    
    Returns:
        Список кадров с найденными совпадениями
    """
    # Проверка существования файлов
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Видео файл не найден: {video_path}")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Шаблон не найден: {template_path}")
    
    # Загрузка шаблона (в оттенках серого для обработки и в цвете для визуализации)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    template_color = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None or template_color is None:
        raise ValueError(f"Не удалось загрузить шаблон: {template_path}")
    
    print(f"Размер шаблона: {template.shape}")
    
    # Инициализация SIFT детектора
    sift = cv2.SIFT_create()
    
    # Вычисление ключевых точек и дескрипторов для шаблона
    kp_template, des_template = sift.detectAndCompute(template, None)
    print(f"Найдено ключевых точек в шаблоне: {len(kp_template)}")
    
    # Инициализация окон для визуализации
    if show_preview:
        cv2.namedWindow('Template', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Detection Result', cv2.WINDOW_NORMAL)
        
        # Отображение шаблона с ключевыми точками
        template_with_kp = template_color.copy()
        cv2.drawKeypoints(template_with_kp, kp_template, template_with_kp, 
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.putText(template_with_kp, f"Keypoints: {len(kp_template)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Template', template_with_kp)
    
    if des_template is None or len(des_template) == 0:
        raise ValueError("Не удалось найти ключевые точки в шаблоне")
    
    # Открытие видео файла
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    
    # Получение параметров видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Параметры видео: {width}x{height}, {fps} FPS, {total_frames} кадров")
    print(f"Параметры детекции: match_threshold={match_threshold}, min_matches={max(4, len(kp_template) // 20)}")
    
    # Инициализация видеописателя (если указан путь для сохранения)
    # Ускоряем видео в 10 раз для быстрого просмотра результата
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        speeded_fps = fps * 2  # Ускоряем в 10 раз
        writer = cv2.VideoWriter(output_path, fourcc, speeded_fps, (width, height))
        print(f"Видео будет сохранено с FPS: {speeded_fps} (исходный FPS: {fps}, ускорение в 10 раз)")
    
    # Инициализация матчера (FLANN или Brute Force)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    frame_count = 0
    matches_found = 0
    matches_window_created = False
    
    print("\nОбработка видео...")
    print("Управление: 'q' - выход, 'пробел' - пауза")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Сохранение исходного кадра для визуализации
        original_frame = frame.copy()
        matches_img = None  # Инициализация переменной для совпадений
        min_matches = max(4, len(kp_template) // 20)  # Адаптивный минимум совпадений
        good_matches = []  # Инициализация для визуализации
        kp_frame = []  # Инициализация для случая отсутствия ключевых точек
        
        # Конвертация кадра в оттенки серого
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Вычисление ключевых точек и дескрипторов для текущего кадра
        kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)
        
        if des_frame is not None and len(des_frame) > 0:
            # Поиск совпадений между шаблоном и кадром
            matches = flann.knnMatch(des_template, des_frame, k=2)
            
            # Фильтрация совпадений по соотношению Лоу (Lowe's ratio test)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < match_threshold * n.distance:
                        good_matches.append(m)
            
            # Если найдено достаточно хороших совпадений
            if len(good_matches) >= min_matches:
                matches_found += 1
                
                # Получение координат совпадающих точек
                src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Вычисление гомографии для определения местоположения объекта
                # Увеличиваем порог RANSAC для более гибкой детекции
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0, maxIters=3000, confidence=0.99)
                
                # Создание кадра результата детекции
                result_frame = original_frame.copy()
                
                if M is not None:
                    # Получение углов шаблона
                    h, w = template.shape
                    corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    
                    # Преобразование углов в координаты кадра
                    transformed_corners = cv2.perspectiveTransform(corners, M)
                    
                    # Рисование рамки вокруг найденного объекта
                    result_frame = cv2.polylines(result_frame, [np.int32(transformed_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
                    
                    # Рисование ключевых точек на кадре
                    result_frame = cv2.drawKeypoints(result_frame, kp_frame, None, 
                                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    
                    # Добавление текста с информацией
                    cv2.putText(result_frame, f"Matches: {len(good_matches)}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(result_frame, f"Frame: {frame_count}/{total_frames}", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(result_frame, "OBJECT FOUND", 
                              (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Создание изображения с совпадениями (side-by-side)
                    matches_img = cv2.drawMatches(
                        template_color, kp_template,
                        original_frame, kp_frame,
                        good_matches[:50], None,  # Показываем первые 50 совпадений
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )
                else:
                    # Если гомография не найдена
                    result_frame = cv2.drawKeypoints(result_frame, kp_frame, None, 
                                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    cv2.putText(result_frame, f"Matches: {len(good_matches)}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(result_frame, f"Frame: {frame_count}/{total_frames}", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    matches_img = cv2.drawMatches(
                        template_color, kp_template,
                        original_frame, kp_frame,
                        good_matches[:50], None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )
                
                # Используем result_frame для сохранения
                frame = result_frame
            else:
                # Если совпадений недостаточно
                result_frame = original_frame.copy()
                cv2.putText(result_frame, f"Frame: {frame_count}/{total_frames}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(result_frame, "Object not found", 
                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(result_frame, f"Matches: {len(good_matches)}/{min_matches}", 
                          (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Отладочная информация
                if frame_count % 30 == 0 and len(good_matches) > 0:
                    print(f"Кадр {frame_count}: найдено {len(good_matches)} совпадений, требуется {min_matches}")
                
                frame = result_frame
        else:
            # Если не найдено ключевых точек в кадре
            result_frame = original_frame.copy()
            cv2.putText(result_frame, f"Frame: {frame_count}/{total_frames}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
            cv2.putText(result_frame, "No keypoints detected", 
                      (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
            
            frame = result_frame
        
        # Создание кадра с шаблоном в углу (для сохранения и визуализации)
        original_display, template_pos = overlay_template(original_frame.copy(), template_color, 
                                                         position='bottom_right', scale=0.15, margin=15)
        cv2.putText(original_display, f"Frame: {frame_count}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(original_display, "ORIGINAL", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Добавляем информацию о совпадениях на исходный кадр
        if des_frame is not None and len(des_frame) > 0:
            if len(good_matches) > 0:
                color = (0, 255, 0) if len(good_matches) >= min_matches else (0, 255, 255)
                cv2.putText(original_display, f"Good matches: {len(good_matches)}", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(original_display, f"Threshold: {match_threshold:.2f}", 
                           (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                # Рисуем связи между ключевыми точками шаблона и кадра
                original_display = draw_keypoint_connections(
                    original_display, template_pos, template.shape[:2],
                    kp_template, kp_frame, good_matches,
                    template_scale=0.15, frame_scale=1.0
                )
            else:
                cv2.putText(original_display, f"Matches: 0", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Визуализация
        if show_preview:
            # Отображение результата детекции
            cv2.imshow('Detection Result', frame)
            
            # Отображение совпадений (если есть)
            if matches_img is not None:
                if not matches_window_created:
                    cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
                    matches_window_created = True
                cv2.imshow('Matches', matches_img)
            
            # Обработка клавиш с замедлением для лучшей видимости (100 мс = ~10 FPS)
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                print("\nОстановка по запросу пользователя (нажата клавиша 'q')")
                break
            elif key == ord(' '):  # Пробел для паузы
                print("Пауза. Нажмите любую клавишу для продолжения...")
                cv2.waitKey(0)
        
        # Сохранение кадра с шаблоном и связями (если указан путь)
        if writer:
            writer.write(original_display)
        
        # Вывод прогресса
        if frame_count % 30 == 0:
            print(f"Обработано кадров: {frame_count}/{total_frames} | Найдено совпадений: {matches_found}")
    
    # Освобождение ресурсов
    cap.release()
    if writer:
        writer.release()
    
    # Закрытие окон визуализации
    if show_preview:
        cv2.destroyAllWindows()
    
    print(f"\nОбработка завершена!")
    print(f"Всего кадров: {frame_count}")
    print(f"Кадров с найденным объектом: {matches_found}")
    
    return matches_found


def main():
    """Основная функция"""
    # Пути к файлам
    video_path = "get.197.025.left.avi"
    template_path = "template.png"
    output_path = "output_detection.mp4"  # Опционально: путь для сохранения результата
    
    try:
        # Выполнение детекции
        matches_count = detect_object_sift(
            video_path=video_path,
            template_path=template_path,
            output_path=output_path,  # Установите None, если не нужно сохранять
            match_threshold=0.6,  # Порог для фильтрации совпадений (0.5-0.7, меньше = более строгий)
            show_preview=True  # Показывать визуализацию в реальном времени
        )
        
        print(f"\nРезультат: объект найден в {matches_count} кадрах")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
