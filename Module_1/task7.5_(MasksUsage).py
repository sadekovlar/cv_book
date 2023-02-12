import cv2
import os
import numpy as np


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

    # create mask for vehicle
    mask = np.zeros(input_image.shape[:2], np.uint8)
    # start point (left_top) & end point (right_bottom)
    vehicle_rect = [(530, 260), (530 + 80, 260 + 65)]
    cv2.rectangle(mask, vehicle_rect[0], vehicle_rect[1],
                  color=255, thickness=-1)
    # apply the mask
    res = cv2.bitwise_and(input_image, input_image, mask=mask)
    cv2.imshow('res', res)

    k = cv2.waitKey()
    cv2.destroyAllWindows()

    # create mask for cross-road
    mask = np.zeros(input_image.shape[:2], np.uint8)
    # 4 points
    pts = np.array([
        [200, 315], [240, 305], [530, 295], [530, 300]
    ], np.int32)
    # not polylines 'cause we should have a filled region
    cv2.fillPoly(mask, [pts], color=255)
    # apply the mask
    res = cv2.bitwise_and(input_image, input_image, mask=mask)
    cv2.imshow('res', res)
    k = cv2.waitKey()
    cv2.destroyAllWindows()
