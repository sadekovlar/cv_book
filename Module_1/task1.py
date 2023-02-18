import cv2
import matplotlib.pyplot as plt

image = cv2.imread("./image.jpg")
cv2.imshow("Hello World", image)
cv2.waitKey()
cv2.destroyAllWindows()

print(image.shape)

print("Height of Image:", int(image.shape[0]), "pixels")
print("Width of Image: ", int(image.shape[1]), "pixels")
print("No of RGB elements: ", int(image.shape[2]), "values")

# Пример того как можно получить (r, g, b) значения для отдельно взятого пикселя.
(b, g, r) = image[0, 0]

print('(r, g, b) values for pixel[0,0]', '\n', "Red: {}, Green: {}, Blue: {}".format(r, g, b))

# Пример использования метода для размытия:

# Использование метода blur (берёт среднее значение)
average_image = cv2.blur(image, (10, 10))

plt.imshow(average_image)
plt.show()

# Использование метода Gaussianblur (Гауссовский метод)

blur = cv2.GaussianBlur(image, (5, 5), 0)

plt.imshow(blur)
plt.show()

# Использование метода medianblur (берёт медианное значение)

median = cv2.medianBlur(image, 5)

plt.imshow(median)
plt.show()

cv2.imwrite("output.jpg", image)
cv2.imwrite("output.png", image)
