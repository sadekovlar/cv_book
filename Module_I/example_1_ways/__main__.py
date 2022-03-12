import cv2

from Module_I.load_calib import CalibReader
from Module_I.example_1_ways.way_estimator import WayEstimator


if __name__ == '__main__':
    par = ['K', 'D', 'r', 't']
    calib_reader = CalibReader(
        file_name=r'../../data/tram/leftImage.yml',
        param=par)
    calib_dict = calib_reader.read()

    path = r'reels.bmp'
    img = cv2.imread(path)
    # todo: зачитать из калиба sz
    img = cv2.resize(img, (960, 540), interpolation = cv2.INTER_AREA)

    way_estimator = WayEstimator(calib_dict, 10)
    img = way_estimator.dray_way(img)

    cv2.namedWindow('reels')
    cv2.imshow('reels', img)
    cv2.waitKey(0)