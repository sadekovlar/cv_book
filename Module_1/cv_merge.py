import cv2

image = cv2.imread("./image.jpg")
b, g, r = cv2.split(image)  # получаем каналы для Blue, Green и Red

cv2.imshow("original", image)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow("blue", b)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow("green", g)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow("red", r)
cv2.waitKey()
cv2.destroyAllWindows()

merged = cv2.merge([b, g, r])  # Производим слияние каналов
cv2.imshow("merged", merged)
cv2.waitKey()
cv2.destroyAllWindows()
