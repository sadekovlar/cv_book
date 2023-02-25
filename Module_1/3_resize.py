import cv2

# load our input image
image = cv2.imread("./image.jpg")

# Let's make our image 3/4 of it's original size
image_scaled = cv2.resize(image, dsize=None, fx=0.75, fy=0.75)
cv2.imshow("image 1", image_scaled)
cv2.waitKey()

# Let's double the size of our image
image_scaled = cv2.resize(image, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
cv2.imshow("image 2", image_scaled)
cv2.waitKey()

# Let's skew the re-sizing by setting exact dimensions
image_scaled = cv2.resize(image, dsize=(900, 300), interpolation=cv2.INTER_AREA)
cv2.imshow("image 3", image_scaled)
cv2.waitKey()
cv2.destroyAllWindows()
