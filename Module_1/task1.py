import cv2

input_ = cv2.imread("./image.jpg")
cv2.imshow("Hello World", input_)
cv2.waitKey()
cv2.destroyAllWindows()

print(input_.shape)

print("Height of Image:", int(input_.shape[0]), "pixels")
print("Width of Image: ", int(input_.shape[1]), "pixels")
print("No of RGB elements: ", int(input_.shape[2]), "values")

cv2.imwrite("output.jpg", input_)
cv2.imwrite("output.png", input_)
