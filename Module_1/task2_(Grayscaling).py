import cv2
import os
import numpy as np


# we can use custom BGR-ratios and get more specific grayscale image
# by default, values from OpenCV docs
# https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray
def grayscale_custom(img_in, blue_ratio=0.114,
                     green_ratio=0.587, red_ratio=0.299):
    if img_in.shape[2] == 3:
        b, g, r = cv2.split(img_in)

        b = np.multiply(b, blue_ratio)
        g = np.multiply(g, green_ratio)
        r = np.multiply(r, red_ratio)
        # for the image to be displayed correctly, it must be converted to uint8
        # for which the pixel brightness is stored in the range [0, 255]
        # for floating-point types, the range is [0, 1]
        img_out = (b + g + r).astype(np.uint8)  # equivalent to cv2.add()
        return img_out
    else:
        print('The input must have 3 channels')


if __name__ == "__main__":
    # check the current working directory (CWD)
    # run examples from root (cv_book folder)
    cwd = os.getcwd()
    # Load our input image
    file_path = os.path.join('data', 'road.png')
    input_image = cv2.imread(file_path)
    if input_image is None:
        print('CWD:', cwd)
        print('Image is empty. Check a file path')
        raise SystemExit(-100)

    # We use cvtColor, to convert to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale', gray_image)

    # Custom grayscale
    gray_img = grayscale_custom(input_image)
    cv2.imshow('Grayscale (Custom)', gray_img)

    cust_gray_img = grayscale_custom(input_image, 0.33, 0.33, 0.33)
    cv2.imshow('Grayscale (Custom-Mean)', cust_gray_img)

    cv2.waitKey()
    cv2.destroyAllWindows()

    # Another method faster method
    # The second argument of 0 makes it greyscale
    img = cv2.imread(file_path, 0)

    cv2.imshow('Grayscale', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
