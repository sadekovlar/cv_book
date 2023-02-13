import cv2

image = cv2.imread("./image.jpg")
height, width = image.shape[:2]  # Первые 2 элемента кортежа shape

start_row = int(height * 0.25)  # int - приведение float к int с округлением в нижную сторону
start_col = int(width * 0.25)

end_row = int(height * 0.80)
end_col = int(width * 0.80)

cropped = image[start_row:end_row, start_col:end_col]

cv2.imshow("original_image", image)
cv2.waitKey()
cv2.imshow("cropped", cropped)
cv2.waitKey()
cv2.destroyAllWindows()
