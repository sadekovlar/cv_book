# the source file was taken from
# https://id.foursquare.com/v/staleywise-gallery/4afeebdbf964a520ac3122e3/photos

import cv2
import os


if __name__ == "__main__":
    # check the current working directory (CWD)
    # run examples from root (cv_book folder)
    cwd = os.getcwd()
    # Load our input image
    file_path = os.path.join('data', 'module_1', 'bin_sample.jpg')
    input_image = cv2.imread(file_path)
    if input_image is None:
        print('CWD:', cwd)
        print('Image is empty. Check a file path')
        raise SystemExit(-100)

    # binarization (expected) works with one-channel images
    gray_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # binarization:
    # 1) each pixel with a brightness *less* than the THRESHOLD value becomes 0
    # 2) each pixel with a b-ness *more* than the THRESHOLD value becomes TARGET
    THRESHOLD = 127  # change a value to obtain a better result
    TARGET = 255
    # use '_' for value skipping
    _, bin_img = cv2.threshold(gray_img, THRESHOLD, TARGET,
                               cv2.THRESH_BINARY)
    _, inv_bin = cv2.threshold(gray_img, THRESHOLD, TARGET,
                               cv2.THRESH_BINARY_INV)

    cv2.imshow('ORIGINAL', gray_img)
    cv2.imshow('BINARY', bin_img)
    cv2.imshow('BINARY_INV', inv_bin)

    k = cv2.waitKey()
    cv2.destroyAllWindows()
