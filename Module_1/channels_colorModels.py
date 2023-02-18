import cv2
import numpy as np

# read image from file
im = cv2.imread("image.jpg")

# resize original image
im = cv2.resize(im, (300, 300))

# split to RGB channels
im_b, im_g, im_r = cv2.split(im)

# concatenate image channels into one image and show
# axis 1 = horisontal concatenate
imgs = np.concatenate((im_b, im_g), axis=1)
imgs = np.concatenate((imgs, im_r), axis=1)

cv2.imshow(imgs)
cv2.waitKey(0)

# color models

# standard opencv color model
bgr_im = im

# rgb color model
rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# hsv color model
hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

# xyz color model
xyz_im = cv2.cvtColor(im, cv2.COLOR_BGR2XYZ)

imgs = np.concatenate((bgr_im, rgb_im, hsv_im, xyz_im), axis=1)

cv2.imshow(imgs)

cv2.waitKey(0)
cv2.destroyAllWindows()
