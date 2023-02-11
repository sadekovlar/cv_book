import cv2
import numpy as np 
input = cv2.imread('image.jpg')
cv2.imshow('Hello World', input)
cv2.waitKey()
cv2.destroyAllWindows()

print(input.shape)

print('Height of Image:', int(input.shape[0]), 'pixels')
print('Width of Image: ', int(input.shape[1]), 'pixels')
print('No of RGB elements: ', int(input.shape[2]), 'values')

cv2.imwrite('output.jpg', input)
cv2.imwrite('output.png', input)