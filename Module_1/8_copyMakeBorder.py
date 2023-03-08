import cv2

IMAGE_PATH = "Module_1/road.png"

image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

BLUE = [0, 0, 255]

# * copuMakeBorder is function that copy source image in the middle of new image and add pixels to border of new image
# *  top, bottom, left and right - parameters specifying
# *  how many pixels in each direction from the source image rectangle to extrapolate
replicate = cv2.copyMakeBorder(
    image, top=50, bottom=50, left=50, right=50, borderType=cv2.BORDER_REPLICATE
)  # * Pixels of border copy last pixels from original image
reflect = cv2.copyMakeBorder(
    image, top=50, bottom=50, left=50, right=50, borderType=cv2.BORDER_REFLECT
)  # * Border will be mirror reflection of the border element
reflect101 = cv2.copyMakeBorder(
    image, top=50, bottom=50, left=50, right=50, borderType=cv2.BORDER_REFLECT_101
)  # * The same, but the last pixels didn't repeat
wrap = cv2.copyMakeBorder(
    image, top=50, bottom=50, left=50, right=50, borderType=cv2.BORDER_WRAP
)  # * Add border with pixels from opposite side
constant = cv2.copyMakeBorder(
    image, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=BLUE
)  # * Border consists constant 'value'

cv2.imshow("replicated", replicate)
cv2.waitKey()

cv2.imshow("reflected", reflect)
cv2.waitKey()

cv2.imshow("reflected", reflect101)
cv2.waitKey()

cv2.imshow("wrapped", wrap)
cv2.waitKey()

cv2.imshow("constant blue", constant)
cv2.waitKey()
cv2.destroyAllWindows()
