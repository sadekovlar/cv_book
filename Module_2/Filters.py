# importing libraries
import cv2
import numpy as np


# 1. Blurring
image = cv2.imread('horses.jpg')

cv2.imshow('original', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 1.1 Averaging blurring
"""
An average filter takes an area of pixels surrounding a central pixel,
averages all these pixels together, and replaces the central pixel with the average.
By taking the average of the region surrounding a pixel, we are smoothing it and replacing 
it with the value of its local neighborhood. This allows us to reduce noise and the level of detail, 
simply by relying on the average.
"""
# averaging blurring
img_blur_av = cv2.blur(image, (9, 9))

cv2.imshow('Averaging', img_blur_av)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 1.2 Gaussian blurring
"""
# Gaussian blurring is similar to average blurring, 
but instead of using a simple mean, we are now using a weighted mean, 
where neighborhood pixels that are closer to the central pixel contribute more “weight” to the average.
"""
# Gaussian blurring
img_blur_gauss= cv2.GaussianBlur(image,(9,9),0)

cv2.imshow('Gauss', img_blur_gauss)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 1.3 Median blurring
"""
# Traditionally, the median blur method has been most effective when removing salt-and-pepper noise. 
Notice that (unlike the averaging method), instead of replacing the central pixel with 
the average of the neighborhood, we instead replace the central pixel with the median of the neighborhood.
The reason median blurring is more effective at removing salt-and-pepper style noise from an image 
is that each central pixel is always replaced with a pixel intensity that exists in the image. 
And since the median is robust to outliers, the salt-and-pepper noise will be less influential to the median 
than another statistical method, such as the average.
"""

# median blurring
img_blur_median = cv2.medianBlur(image,9)

cv2.imshow('Median', img_blur_median)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 1.4 Bilateral blurring
"""
# This method is able to preserve edges of an image, while still reducing noise.  
Bilateral blurring accomplishes this by introducing two Gaussian distributions. 
The largest downside to this method is that it is considerably slower than its averaging, Gaussian, 
and median blurring counterparts.
"""

# bilateral blurring
img_blur_bil = cv2.bilateralFilter(image,  9, 75, 75, cv2.BORDER_DEFAULT)

cv2.imshow('Bilateral', img_blur_bil)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 1.4 Usage
"""
# Smoothing and blurring is one of the most important preprocessing steps in all of computer vision 
and image processing. By smoothing an image prior to applying techniques such as edge detection or
thresholding we are able to reduce the amount of high-frequency content, such as noise and edges.
Very often, blurring of images is aimed at imitating myopia, for a visual concept of what a person will see.
"""

# 2.Image filtering using convolution
"""
# While dealing with images in Image Processing, filter2D() function is used to change the pixel intensity 
value of an image based on the surrounding pixel intensity values. This method can enhance or remove certain
features of an image to create a new image.

More formally, filter2D() function convolves an image with the kernel which results in an image becoming 
blur or sharpen and enhances the image features.
"""

# 2.1 Emboss filter

# creating the kernel(2d convolution matrix)
kernel = np.array([
  [-2, -1, 0],
  [-1, 1, 1],
  [0, 1, 2]
])
  
# applying the filter2D() function
img_emb = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

cv2.imshow('Emboss filter', img_emb)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2.2 Sobel filter

# creating the kernel(2d convolution matrix)
kernel = np.array([
  [-1, 0, 1],
  [-2, 0, 2],
  [-1, 0, 1]
])
  
# applying the filter2D() function
img_sobel = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

cv2.imshow('Sobel filter', img_sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2.3 Outline edge detection

# creating the kernel(2d convolution matrix)
kernel = np.array([
  [-1, -1, -1],
  [-1, 8, -1],
  [-1, -1, -1]
])
  
# applying the filter2D() function
img_out = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)


cv2.imshow('Outline edge detection', img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
OpenCV filter2D() function is used in python to manipulate images. 
Everyone can write the kernel themselves to achieve the desired result.
"""