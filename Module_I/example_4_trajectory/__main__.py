import cv2

from Module_I.load_calib import CalibReader
from Module_I.example_4_trajectory.trajectory_estimator import TrajectoryEstimator


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

    traj_estimator = TrajectoryEstimator(calib_dict=calib_dict,
                                         height=1.6,
                                         wight=1.6,
                                         length=25,
                                         depth=8)
    img = traj_estimator.dray_trajectory(img)

    cv2.namedWindow('reels')
    cv2.imshow('reels', img)
    cv2.waitKey(0)