import cv2

image = cv2.imread("./Module_1/img/image.jpg")
cv2.imshow("Hello World", image)
cv2.waitKey()
cv2.destroyAllWindows()

print(image.shape)

print("Height of Image:", int(image.shape[0]), "pixels")
print("Width of Image:", int(image.shape[1]), "pixels")
print("No of RGB elements:", int(image.shape[2]), "values")

cv2.imwrite("./Module_1/img/output.jpg", image)
cv2.imwrite("./Module_1/img/output.png", image)
