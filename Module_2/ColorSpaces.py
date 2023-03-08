# import libraries
import cv2
import numpy as np

# 1 - BGR color-space
image = cv2.imread('Module_1a/img/map.jpg')

# BGR Values for the first 0,0 pixel
B, G, R = image[10, 50]
print(B, G, R)
print(image.shape)

# 2 - Grayscale color-space
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray_img.shape)
print(gray_img[10, 50])

# 3 - HSV color-space

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

# 4 - LAB color-space
# (L – Lightness, a – color component ranging from Green to Magenta,b – color component ranging from Blue to Yellow)
image = cv2.imread('Module_1a/img/map.jpg')
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("L*a*b*", lab)

for (name, chan) in zip(("L*", "a*", "b*"), cv2.split(lab)):
	cv2.imshow(name, chan)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 5 - YCrCb color-space
# Y – Luminance or Luma component obtained from RGB after gamma correction
# Cr = R – Y ( how far is the red component from Luma )
# Cb = B – Y ( how far is the blue component from Luma )

image = cv2.imread('Module_1a/img/map.jpg')
ycb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
cv2.imshow("YCrCb", ycb)


cv2.waitKey(0)
cv2.destroyAllWindows()

# 6 - YUV color-space
# Y - luminance (brightness), U and V define the blue projection and red projection
image = cv2.imread('Module_1a/img/map.jpg')
yuv = cv2.cvtColor(image, cv2.cv2.COLOR_BGR2YUV)
cv2.imshow("YUV", yuv)


cv2.waitKey(0)
cv2.destroyAllWindows()

# 7 - individual channels in an RGB image

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

# 8
B, G, R = cv2.split(image)

# create a matrix of zeros with dimensions of the image h x w
zeros = np.zeros(image.shape[0:2], dtype = "uint8")

cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))

cv2.waitKey(0)
cv2.destroyAllWindows()

image.shape[0:2]