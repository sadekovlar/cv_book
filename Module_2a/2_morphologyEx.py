# import libraries
import cv2
import numpy as np


image = cv2.imread('./image.jpg')

# structuring element or kernel
kernel = np.ones((5,5),np.uint8)

# 1 - opening
# is just another name of erosion followed by dilation
# it is useful in removing noise, as we explained above
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# 2 - closing
# is reverse of Opening, Dilation followed by Erosion
# it is useful in closing small holes inside the foreground objects, or small black points on the object
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


# 3 - morphological gradient
# it is the difference between dilation and erosion of an image
# the result will look like the outline of the object
gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


# 4 - top hat
# it is the difference between input image and Opening of the image. Below example is done for a 9x9 kernel
tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

# 5 - black hat
# it is the difference between the closing of the input image and input image
blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

cv2.imshow('Input', image)
cv2.waitKey(0)
cv2.imshow('Opening', opening)
cv2.waitKey(0)
cv2.imshow('Closing', closing)
cv2.waitKey(0)
cv2.imshow('Gradient', gradient)
cv2.waitKey(0)
cv2.imshow('Top Hat', tophat)
cv2.waitKey(0)
cv2.imshow('Black Hat', blackhat)
cv2.waitKey(0)

cv2.destroyAllWindows()