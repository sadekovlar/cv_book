import cv2
import numpy as np

# Рассматриваемые задачи:
# 1. Возможные способы размытия изображений
# 2. Выделение границ и работа с ними

img = cv2.imread('image.jpg')

# Размытие изображения

# OpenCV предоставляет четыре основных типа методов размытия.
# Первый способ - cv2.blur() - размытие осуществляется за счет свертки. В параметрах функции указывается размер свертки

blur = cv2.blur(img,(50,50))
cv2.imshow('blur', blur)

# Второй способ - гаусовское размытие - cv2.GaussianBlur() - используется ядро ​​Гаусса 
# Аналогично указываются параметры ширины и высоты ядра, которые должны быть положительными и нечетными.
# Также указывается параметр стандартного отклонения

blur_gauss = cv2.GaussianBlur(img,(51,51),5)
cv2.imshow('blur_gauss', blur_gauss)

# Третий способ - cv2.medianBlur() - центральный пиксель ядра заменяется на его медиану
# Такой способ эффективней против шумов

blur_median = cv2.medianBlur(img,5)
cv2.imshow('blur_median', blur_median)


# Четвертая функция - двусторонний фильтр -  cv2.bilateralFilter()  
# Также использует Гаусовское размытие, но с двуми фильтрами, второй для разности интенсивности пикселей
# Такой способ размытия лучше сохраняет границы изображения

blur_gauss_bi = cv2.bilateralFilter(img,9,75,75)
cv2.imshow('blur_gauss_bi ', blur_gauss_bi )

cv2.waitKey()
cv2.destroyAllWindows()

#  Выделение границ изображения

# Функция для отрисовки контуров - cv2.Canny
# В параметрах изображения и пороги для интенсивности пикселей

edges = cv2.Canny(img,100,200)
cv2.imshow('edges', edges)
cv2.waitKey()
cv2.destroyAllWindows()

# Для выделения контуров используется функция cv2.findContours()
# Результат записывается в две переменные - в первую список точек контура, а во вторую их иерархия
# В функции cv.findContours() есть три аргумента, первый — исходное изображение, второй — режим поиска контура, третий — метод аппроксимации контура

img2 = cv2.imread('image.jpg', 0)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Можем посмотреть как это хранится 
print(contours)
print(hierarchy)

# По полученным точкам можно отрисовать контур изображения на нем самом, например другого цвета или толщины
# Для этого используется  cv2.drawContours - на вход подается изображение, список точек, индекс контуров (-1 - все контуры) цвет и толщина

contours = cv2.drawContours(img, contours, -1, (0,255,0), 3)
cv2.imshow('contours', contours)
cv2.waitKey()
cv2.destroyAllWindows()

# Также на примере контуров можно посмотреть как работают морфологические функции - cv2.dilate() и cv2.erode()
# Для этих функций нужно задать ядро

kernel = np.ones((3, 3), 'uint8')

# cv2.dilate() - расширение изображения
dilate_img = cv2.dilate(edges, kernel, iterations=1)
cv2.imshow('Dilated Image', dilate_img)
# cv2.erode() - эрозия изображения
erode_img = cv2.erode(dilate_img, kernel, cv2.BORDER_REFLECT, iterations=1)
cv2.imshow('Eroded Image', erode_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

