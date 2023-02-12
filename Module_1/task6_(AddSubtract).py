import cv2
import numpy as np
import os


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

    height = input_image.shape[0]
    width = input_image.shape[1]
    depth = input_image.shape[2]

    M = np.ones([height, width, depth], dtype="uint8")*100

    added = cv2.add(input_image, M)

    cv2.imshow('added', added)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    M = np.ones(input_image.shape, dtype="uint8") * 75
    subtracted = cv2.subtract(M, input_image)
