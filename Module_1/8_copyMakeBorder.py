import cv2

IMAGE_PATH = './Module_1/img/image.jpg'

image = cv2.imread(IMAGE_PATH)

BLUE = [255, 0, 0]

# copyMakeBorder is a function that copies the source image to the middle of the new image and adds pixels to the border
# top, bottom, left, right - parameters specifying how many pixels in each direction from the image to extrapolate

# Pixels of the border copy last pixels from original image
replicate = cv2.copyMakeBorder(image, top=50, bottom=50, left=50, right=50, borderType=cv2.BORDER_REPLICATE)

# Border is a mirror reflection of the border element
reflect = cv2.copyMakeBorder(image, top=50, bottom=50, left=50, right=50, borderType=cv2.BORDER_REFLECT)

# The same, but the last pixels don't repeat
reflect101 = cv2.copyMakeBorder(image, top=50, bottom=50, left=50, right=50, borderType=cv2.BORDER_REFLECT_101)

# Adds border with pixels from opposite side
wrap = cv2.copyMakeBorder(image, top=50, bottom=50, left=50, right=50, borderType=cv2.BORDER_WRAP)

# Border consists of the constant color value
constant = cv2.copyMakeBorder(image, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=BLUE)

cv2.imshow('replicated', replicate)
cv2.waitKey()

cv2.imshow('reflected', reflect)
cv2.waitKey()

cv2.imshow('reflected', reflect101)
cv2.waitKey()

cv2.imshow('wrapped', wrap)
cv2.waitKey()

cv2.imshow('constant blue', constant)
cv2.waitKey()
cv2.destroyAllWindows()
