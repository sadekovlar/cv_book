import cv2
import numpy as np

image = cv2.imread('image.jpg')
height, width = image.shape[:2]

#cv2.getRotationMatrix2D(rotation_center_x, rotation_center_y, angle of rotation, scale)
rotation_matrix = cv2.getRotationMatrix2D((width/2,height/2), 20, .9)
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

cv2.imshow('image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# other option to rotate - transpose
img = cv2.imread('image.jpg')

rotated_image = cv2.transpose(img)

cv2.imshow('Rotated Image - Method 2', rotated_image)
cv2.waitKey()
cv2.destroyAllWindows()

# last option - flips
flipped = cv2.flip(image, 1)
cv2.imshow('Horizontal flip', flipped)
flipped2 = cv2.flip(image, 0)
cv2.imshow('Vertical flip', flipped2)
flipped3 = cv2.flip(image, -1)
cv2.imshow('Diagonal flip', flipped3)

cv2.waitKey()
cv2.destroyAllWindows()

