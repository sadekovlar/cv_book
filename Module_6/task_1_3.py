import cv2
import numpy as np

# функция отрисовки сетки и извлечения
# соответствующего оптического потока для каждой точки сетки 
# и отрисовки линий и точек для визуализации оптического потока.
def draw_flow(im,flow,step=16):
    h,w = im.shape[:2]
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1).astype(int)
    print(y)
    print(x)
    fx,fy = flow[y,x].T
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)
    vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for (x1,y1),(x2,y2) in lines:
        cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.circle(vis,(x1,y1),1,(0,255,0), -1)
    return vis

# функция преобразования оптического потока в цветовое представление HSV
def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    v = np.sqrt (fx*fx + fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def move_direction(flow, frame):
    # разделение оптического потока на компоненты по осям X и Y 
    x, y = flow[..., 0],flow[..., 1]

    # вычисление средних значений компонентов движения
    mean_x, mean_y = np.mean(x),np.mean(y)

    # определение направления движения
    if abs(mean_x) > abs(mean_y):
        if mean_x > 0:
            movement_direction = "Motion Direction: Left"
        else:
            movement_direction = "Motion Direction: Right"
    else:
        if abs(mean_y) < 0.5:
            movement_direction = "Motion Direction: Don't Moving"
        elif mean_y > 0:
            movement_direction = "Motion Direction: Forward"
        else:
            movement_direction = "Motion Direction: Back"

    # визуализация движения
    start = (frame.shape[1] // 2, frame.shape[0] // 2)
    end = (frame.shape[1] // 2 + int(100 * mean_x), frame.shape[0] // 2 + int(100 * mean_y))
    move_dir = cv2.arrowedLine(frame, end, start, (0, 255, 0), 5)
    cv2.putText(move_dir, movement_direction,(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return move_dir

# функция нахождения неподвижных областей через пороговые значения магнитуды и угла оптического потока
def find_static_objects(flow, mag_threshold, ang_threshold):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    static_mask = np.logical_and(mag > mag_threshold, ang < ang_threshold)
    return static_mask


cap = cv2.VideoCapture('../data/city/trm.169.008.avi')
ret,im = cap.read()
prev_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
while True:
    ret,im = cap.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    
    # Вычисление оптического потока с помощью алгоритма Фарнебака
    flow = cv2.calcOpticalFlowFarneback(prev_gray,gray,None,0.5,3,15,3,5,1.2,0)   
    prev_gray = gray

    mag_threshold = 0.7  # Пороговое значение магнитуды для определения неподвижных объектов
    ang_threshold = 3 # Пороговое значение угла для определения неподвижных объектов
    static_mask = find_static_objects(flow, mag_threshold, ang_threshold)
    
    # создание зеленой маски для неподвижных областей
    alpha = 0.5
    overlay = im.copy()
    vis = draw_flow(gray, flow)
    vis[static_mask] = (0, 255, 0)
    cv2.addWeighted(vis, alpha, overlay, 1 - alpha, 0, vis)
    cv2.imshow('Optical flow',vis)
    cv2.imshow('HSV',draw_hsv(flow))
    cv2.imshow('Move_dir', move_direction(flow,im))
    if cv2.waitKey(10) == 27:
        break
                