# this an affine transform that simply shifts the position of an image
# use cv2.warpAffine to implement these transformations

import cv2
import numpy as np

image = cv2.imread('image.jpg')

# store height and width of the image
height, width = image.shape[:2]

quarter_height, quarter_width = height/4, width/4

#       | 1 0 Tx |
#  T  = | 0 1 Ty |

# T is our translation matrix
T = np.float32([[0, 1, quarter_height], [1, 0, quarter_width]])

# using warpAffine to transform the image using the matrix, T
img_translation = cv2.warpAffine(image, T, (width, height))
cv2.imshow('Translation', img_translation)
cv2.waitKey()
cv2.destroyAllWindows()

# look at T
print (T)