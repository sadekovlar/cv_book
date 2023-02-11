import cv2
import numpy as np

# Create a black image
image = np.zeros((512, 512, 3), np.uint8)
# Can we make this in black and white?
image_bw = np.zeros((512,512), np.uint8)
cv2.imshow("Black Rectangle (Color)", image)
cv2.imshow("Black Rectangle (B&W)", image_bw)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Draw a diagonal blue line of thickness of 5 pixels
image = np.zeros((512,512,3), np.uint8)
cv2.line(image, (10,10), (500,500), (100,100,100), 10)
cv2.imshow('drawn' , image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Draw a Rectangle in
image = np.zeros((512,512,3), np.uint8)
cv2.rectangle(image,(100,100), (500,500), (123,123,123),-1)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


image = np.zeros((512,512,3),np.uint8)
cv2.circle(image, (350,350), 100, (200,200,200),5)
cv2.imshow('circle',image)
cv2.waitKey(0)
cv2.destroyAllWindows()


image = np.zeros((512,512,3), np.uint8)
pts = np.array([[10,50], [100,200], [200,250], [120,400]], np.int32)
cv2.polylines(image, [pts], True, (0,0,200), 101)
cv2.imshow('polygonm', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


image = np.zeros((512,512,3), np.uint8)
cv2.putText(image, 'Hello world', (100,100), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (100,160,0), 2)
cv2.circle(image, (400,400), 50, (100,200,255),-1)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()