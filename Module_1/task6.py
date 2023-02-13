import cv2
import numpy as np

image = cv2.imread('image.jpg')
height = image.shape[0]
width = image.shape[1]
depth  = image.shape[2]

M = np.ones([height,width, depth], dtype="uint8")*100

added = cv2.add(image, M)

cv2.imshow('added', added)
cv2.waitKey(0)
cv2.destroyAllWindows()

subtracted = cv2.subtract(M, image)
M = np.ones(image.shape, dtype = "uint8") * 75 