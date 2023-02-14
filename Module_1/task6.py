import cv2
import numpy as np

image = cv2.imread('image.jpg')
height = image.shape[0]
width = image.shape[1]
depth  = image.shape[2]

M = np.ones([height,width, depth], dtype="uint8")*100

added = cv2.add(image, M)
print(image[0,0])
cv2.imshow('added', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

swap =np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
], dtype="uint8")

#lets swap colors!
weird_violet = image.dot(swap)
cv2.imshow('weird_violet', weird_violet)
cv2.waitKey(0)
cv2.destroyAllWindows()

#and another way
weird_acid_blue = weird_violet.dot(swap)
cv2.imshow('weird_acid_blue', weird_acid_blue)
cv2.waitKey(0)
cv2.destroyAllWindows()

#third time
normal = weird_acid_blue.dot(swap)
cv2.imshow('normal', normal)
cv2.waitKey(0)
cv2.destroyAllWindows()

