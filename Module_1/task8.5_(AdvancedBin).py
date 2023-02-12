# the source file was taken from
# https://id.foursquare.com/v/staleywise-gallery/4afeebdbf964a520ac3122e3/photos

import cv2
import numpy as np
import os


def get_win_median(arr, anchor, win_size):
    # we need padding 'cause we out of bounds
    # e.g. point = (0, 0) & win size = 3. the value in (-1, -1) not defined
    # we will just copy the value of pixel which is a border
    carry = win_size // 2
    arr_with_borders = cv2.copyMakeBorder(arr, carry, carry, carry, carry,
                                          cv2.BORDER_REPLICATE)
    temp_arr = arr_with_borders[anchor[0] : anchor[0] + win_size,
                                anchor[1] : anchor[1] + win_size]
    return np.median(temp_arr)


# median windowed binarization
# consume a lot of time for sample image (about 15-30 secs)
def threshold_custom(img_in, max_val, win_size, val=0):
    rows, cols = img_in.shape[:2]
    img_out = np.zeros(img_in.shape[:2], np.uint8)

    for i in range(rows):
        for j in range(cols):
            median = get_win_median(img_in, (i, j), win_size)
            img_out[i, j] = max_val if img_in[i, j] > (median - val) else 0

    return img_out


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

    gray_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    THRESHOLD = 127  # value not used in the Otsu algorithm
    TARGET = 255
    _, bin_image = cv2.threshold(gray_img, THRESHOLD, TARGET,
                                 cv2.THRESH_BINARY)
    # THRESH_OTSU flag allows us to use the Otsu's algorithm to estimate
    # the "best" threshold
    # for more info, please, check
    # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html#otsu:~:text=How%20does%20Otsu%27s%20Binarization%20work%3F
    best_threshold, bin_img = cv2.threshold(gray_img, THRESHOLD, TARGET,
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Best bin threshold: ", best_threshold)

    cv2.imshow('ORIGINAL', gray_img)
    cv2.imshow('BINARY (Otsu)', bin_img)
    cv2.imshow('BINARY', bin_image)

    k = cv2.waitKey()
    cv2.destroyAllWindows()

    # Also we can use adaptive/windowed binarization
    # Only two flags are available: Mean & Gaussian
    # for more info, please, check
    # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html#:~:text=image-,Adaptive%20Thresholding,-In%20the%20previous

    # you can change two values below to obtain different binary images
    window_size = 11  # !only odd number
    const_val = 2.0

    mean = cv2.adaptiveThreshold(gray_img, TARGET,
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, window_size, const_val)
    gauss = cv2.adaptiveThreshold(gray_img, TARGET,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, window_size, const_val)

    cv2.imshow('ORIGINAL', gray_img)
    cv2.imshow('MEAN', mean)
    cv2.imshow('GAUSS', gauss)

    k = cv2.waitKey()
    cv2.destroyAllWindows()

    # consume a lot of time for sample image (about 15-30 secs)
    ready_to_wait = False
    if ready_to_wait:
        th_img = threshold_custom(gray_img, 255, 5, 1)

        cv2.imshow('gray_img', gray_img)
        cv2.imshow('threshold_custom', th_img)

        k = cv2.waitKey()
        cv2.destroyAllWindows()
