# we will search for color by coordinates, this may be useful for design

# importing libraries
import cv2

# uploading an image
image = cv2.imread('111.jpg')

cv2.imshow('original', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# see the size of the image
print(image.shape)

# find the color by coordinates
(b, g, r) = image[100, 0]
print("Красный: {}, Зелёный: {}, Синий: {}".format(r, g, b))