import cv2
import numpy as np

# Load the image
img = cv2.imread('../data/road.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a threshold to the image
thresh_1 = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]
thresh_2 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
thresh_3 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

# Define foreground and background structuring elements
f_kernel = np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]], np.uint8)
b_kernel = np.array([[1, 1, 1], [0, 0, 1], [1, 0, 1]], np.uint8)

# Apply hit-and-miss transformation to the image
hitmiss_1 = cv2.morphologyEx(thresh_1, cv2.MORPH_HITMISS, f_kernel, b_kernel, iterations=1)
hitmiss_2 = cv2.morphologyEx(thresh_2, cv2.MORPH_HITMISS, f_kernel, b_kernel, iterations=1)
hitmiss_3 = cv2.morphologyEx(thresh_3, cv2.MORPH_HITMISS, f_kernel, b_kernel, iterations=1)

# Display the original image and the transformed image
cv2.imshow('Original Image', img)
cv2.waitKey(0)
cv2.imshow('Hit-and-Miss Image', hitmiss_1)
cv2.waitKey(0)
cv2.imshow('Hit-and-Miss Image', hitmiss_2)
cv2.waitKey(0)
cv2.imshow('Hit-and-Miss Image', hitmiss_3)
cv2.waitKey(0)
cv2.destroyAllWindows()
