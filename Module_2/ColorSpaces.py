# import libraries
import cv2
import numpy as np

# 1 - remember about OpenCV's RGB is that it's BGR
image = cv2.imread('Module_1a/img/map.jpg')

# BGR Values for the first 0,0 pixel
B, G, R = image[10, 50]
print(B, G, R)
print(image.shape)

# 2 - what happens when we convert it to grayscale
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray_img.shape)
print(gray_img[10, 50])

# 3 - useful color space is HSV

#H: 0 - 180, S: 0 - 255, V: 0 - 255
image = cv2.imread('Module_1a/img/map.jpg')

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

print(image.shape)

cv2.imshow('HSV image', hsv_image)
cv2.imshow('Hue channel', hsv_image[:, :, 0])
cv2.imshow('Saturation channel', hsv_image[:, :, 1])
cv2.imshow('Value channel', hsv_image[:, :, 2])

cv2.waitKey()
cv2.destroyAllWindows()


# 4 - individual channels in an RGB image

image = cv2.imread('image.jpg')

# openCV's 'split' function splites the image into each color index
B, G, R = cv2.split(image)

print(B.shape)

cv2.imshow("Original", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Red", R)
cv2.imshow("Green", G)
cv2.imshow("Blue", B)
cv2.waitKey(0)
cv2.destroyAllWindows()

# let's re-make the original image,
merged = cv2.merge([B, G, R])
cv2.imshow("Merged", merged)

print(merged - image)

# let's amplify a particular color
merged = cv2.merge([B, G, R+30])
cv2.imshow("Merged with Blue Amplified", merged)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 5
B, G, R = cv2.split(image)

# create a matrix of zeros with dimensions of the image h x w
zeros = np.zeros(image.shape[0:2], dtype = "uint8")

cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))

cv2.waitKey(0)
cv2.destroyAllWindows()

image.shape[0:2]