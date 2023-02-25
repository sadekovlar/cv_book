import cv2

# Load our input image
image = cv2.imread("./image.jpg")
cv2.imshow("Original", image)
cv2.waitKey()

# We use cvtColor, to convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Grayscale", gray_image)
cv2.waitKey()
cv2.destroyAllWindows()


# Another method faster method
# The third argument of 0 makes it greyscale
img = cv2.imread("./image.jpg", 0)

cv2.imshow("Grayscale", img)
cv2.waitKey()
cv2.destroyAllWindows()
