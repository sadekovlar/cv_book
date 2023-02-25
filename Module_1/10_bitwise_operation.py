import cv2
import numpy as np

# If you're wondering why only two dimensions, well this is a grayscale image,
# if we doing a colored image, we'd use
# rectangle = np.zeros((300, 300, 3), np.uint8)

# making a square

square = np.zeros((300, 300), np.uint8)
cv2.rectangle(square, pt1=(50, 50), pt2=(250, 250), color=255, thickness=-1)
cv2.imshow("square", square)

ellipse = np.zeros((300, 300), np.uint8)
cv2.ellipse(ellipse, center=(150, 150), axes=(150, 150), angle=30, startAngle=0, endAngle=120, color=255, thickness=-1)
cv2.imshow("ellpise", ellipse)
cv2.waitKey()
cv2.destroyAllWindows()

# Shows only where they intersect
And = cv2.bitwise_and(square, ellipse)
cv2.imshow("AND", And)
cv2.waitKey()

# Shows where either square or ellipse is
bitwiseOr = cv2.bitwise_or(square, ellipse)
cv2.imshow("OR", bitwiseOr)
cv2.waitKey()

# Shows where either exist by itself
bitwiseXor = cv2.bitwise_xor(square, ellipse)
cv2.imshow("XOR", bitwiseXor)
cv2.waitKey()

# Shows everything that isn't part of the square
bitwiseNot_sq = cv2.bitwise_not(square)
cv2.imshow("NOT - square", bitwiseNot_sq)
cv2.waitKey()

bitwiseNot_el = cv2.bitwise_not(ellipse)
cv2.imshow("NOT - ellipse", bitwiseNot_el)
cv2.waitKey()
# Notice the last operation inverts the image totally

cv2.destroyAllWindows()
