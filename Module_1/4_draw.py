import cv2
import numpy as np

# Create a black image
image = np.zeros((512, 512, 3), np.uint8)
# Can we make this in black and white?
image_bw = np.zeros((512, 512), np.uint8)
cv2.imshow("Black Rectangle (Color)", image)
cv2.imshow("Black Rectangle (B&W)", image_bw)
cv2.waitKey()
cv2.destroyAllWindows()

# Draw a diagonal blue line of thickness of 5 pixels
image = np.zeros((512, 512, 3), np.uint8)
cv2.line(image, pt1=(10, 10), pt2=(500, 500), color=(255, 50, 50), thickness=10)
cv2.imshow("blue line", image)
cv2.waitKey()
cv2.destroyAllWindows()


# Draw a Rectangle in
image = np.zeros((512, 512, 3), np.uint8)
cv2.rectangle(image, pt1=(100, 100), pt2=(500, 500), color=(123, 123, 123), thickness=-1)
cv2.imshow("rectangle", image)
cv2.waitKey()
cv2.destroyAllWindows()

# Draw circle
image = np.zeros((512, 512, 3), np.uint8)
cv2.circle(image, center=(350, 350), radius=100, color=(200, 200, 200), thickness=5)
cv2.imshow("circle", image)
cv2.waitKey()
cv2.destroyAllWindows()

# Draw polygon
image = np.zeros((512, 512, 3), np.uint8)
pts = np.array([[10, 50], [100, 200], [200, 250], [120, 400]], np.int32)
cv2.polylines(image, pts=[pts], isClosed=True, color=(0, 0, 200), thickness=101)
cv2.imshow("polygon", image)
cv2.waitKey()
cv2.destroyAllWindows()

# Text
image = np.zeros((512, 512, 3), np.uint8)
cv2.putText(
    image,
    text="Hello world",
    org=(100, 100),
    fontFace=cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
    fontScale=1,
    color=(100, 160, 0),
    thickness=2,
)
cv2.circle(image, center=(400, 400), radius=50, color=(100, 200, 255), thickness=-1)
cv2.imshow("text", image)
cv2.waitKey()
cv2.destroyAllWindows()
