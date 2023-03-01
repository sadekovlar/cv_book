import cv2

image = cv2.imread("../data/road.png")
height, width = image.shape[:2]

# * pyrDown - увеличение до Х:Y
# * pyrUp - уменьшение до Х:Y
cv2.imshow("original", image)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow("pyrUp, height * 2, width * 2", cv2.pyrUp(image, dstsize=(width * 2, height * 2)))
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow("pyrDown, height // 2, width // 2", cv2.pyrDown(image, dstsize=(width // 2, height // 2)))
cv2.waitKey()
cv2.destroyAllWindows()

# cv2.imshow("pyrDown, height // 3, width // 3", cv2.pyrDown(image, dstsize=(width // 3, height // 3)))
# cv2.waitKey()
# cv2.destroyAllWindows()
