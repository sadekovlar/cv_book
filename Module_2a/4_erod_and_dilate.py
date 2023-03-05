import numpy as np
import cv2

img = cv2.imread("C:/Users/22354/Downloads/picture.jpg")

# Измнение текстуры изображения 
kernel = np.ones((5, 5), np.uint8)

# cv2.erode(src, kernel, dst,anchor,iterations,borderType,borderValue)
# kernel - this is the matrix that is used in the calculations
# So the thickness or size of the foreground object decreases or simply the white region decreases in the image
img_erosion = cv2.erode(img, kernel, iterations=1)

#Increases the object area
#Used to accentuate features
img_dilation = cv2.dilate(img, kernel, iterations=1)


cv2.imshow('Input', img)
cv2.waitKey(0)
cv2.imshow('Erosion', img_erosion)
cv2.waitKey(0)
cv2.imshow('Dilation', img_dilation)
cv2.waitKey(0)
  
cv2.destroyAllWindows()