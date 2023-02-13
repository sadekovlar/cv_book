import cv2
import numpy as np

image = cv2.imread('images/input.jpg')
height, width = image.shape[:2]

start_row = int(height*.25)
start_col = int(width*.25)

end_row = int(height*.80)
end_col = int(width*.80)

cropped = image[start_row:end_row, start_col:end_col]

cv2.imshow("original_image", image)
cv2.waitKey(0)
cv2.imshow("cropped", cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()