import cv2

import src.detection_flares as df

SHOW_SUPPRESS_LIGHTNESS = True
SHOW_SRC = True
SHOW_DRAW_MASK = True
SHOW_MASK = False

HARD_HORIZON_LINE_PX = 170

if __name__ == "__main__":
    # для удобного расположения окон
    cv2.namedWindow("Src")
    cv2.namedWindow("Mask")
    cv2.namedWindow("Draw_mask")
    cv2.namedWindow("Suppress_lightness")
    cv2.moveWindow("Src", 5, 5)
    cv2.moveWindow("Mask", 965, 5)
    cv2.moveWindow("Draw_mask", 5, 545)
    cv2.moveWindow("Suppress_lightness", 965, 545)
    vid_capture = cv2.VideoCapture('./data/processing/trm.174.007.avi')
    if not vid_capture.isOpened():
        print("Ошибка открытия видеофайла")
    else:
        file_count = 0
        while vid_capture.isOpened():
            ret, frame = vid_capture.read()
            if ret:
                # получение маски бликов
                mask = df.get_mask_of_glares(frame, HARD_HORIZON_LINE_PX)
                if SHOW_MASK:
                    cv2.imshow('Mask', mask)
                if SHOW_SRC:
                    cv2.imshow('Src', frame)
                if SHOW_DRAW_MASK:
                    res_dm = df.draw_glares(frame, mask)
                    cv2.imshow('Draw_mask', res_dm)
                if SHOW_SUPPRESS_LIGHTNESS:
                    res_sl = df.suppress_lightness(frame, mask, 0.94)  # можно поиграться с 3им параметром
                    cv2.imshow('Suppress_lightness', res_sl)
                key = cv2.waitKey(20)
                if (key == ord('q')) or key == 27:
                    break
            else:
                break
    # Освободить объект захвата видео
    vid_capture.release()
    cv2.destroyAllWindows()
