import numpy as np
import cv2

img = cv2.imread("C:/Users/22354/Downloads/picture.jpg")


# Функция thresgold принимает только чб изображения
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.threshold(source, thresholdValue, maxVal, thresholdingTechnique) 
# source - our image
# 110 - пороговое значение, все пиксели которые имеют значение больше заменяются на 255, что меньше на 130
# 255 - максимальное значение, которое может быть присвоено пикселю
# 130 -  тип применяемого порогового значения или минимальное значение
ret, img_threshold = cv2.threshold(gray_image, 110, 255, 130)
print(ret)
cv2.imshow("image gray", img_threshold)
cv2.waitKey(0)

# То что выше порога заменеятся на 255, в противном случаезаменяется на 0 (черный).
ret, img_threshold = cv2.threshold(gray_image, 110, 255, cv2.THRESH_BINARY)
cv2.imshow("image gray", img_threshold)
cv2.waitKey(0)

# Инверсия от cv2.THRESH_BINARY.
ret, img_threshold = cv2.threshold(gray_image, 110, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("image gray", img_threshold)
cv2.waitKey(0)

# Если интенсивность больше порога, то оно усекается до порогового, остальные остаются прежними
ret, img_threshold  = cv2.threshold(gray_image,40,255, cv2.THRESH_TRUNC )
cv2.imshow("image gray", img_threshold)
cv2.waitKey(0)

# Все пиксели с интенсивностью ниже порогового становятся 0
ret, img_threshold  = cv2.threshold(gray_image,40,255, cv2.THRESH_TOZERO )
cv2.imshow("image gray", img_threshold)
cv2.waitKey(0)

# Инверсия от cv2.THRESH_TOZERO.
ret, img_threshold  = cv2.threshold(gray_image,40,255, cv2.THRESH_TOZERO_INV )
cv2.imshow("image gray", img_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

