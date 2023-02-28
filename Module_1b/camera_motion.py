import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

# path to the data used for calibration
DATA_DIR = '../data/calib'

# checkerboard parameters
BOARD_DIM = (7, 7)
BOARD_SIZE = 100

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# defining the world coordinates for 3D points
obj_ps = np.zeros((1, BOARD_DIM[0] * BOARD_DIM[1], 3), np.float32)
obj_ps[0, :, :2] = np.mgrid[0:BOARD_DIM[0], 0:BOARD_DIM[1]].T.reshape(-1, 2) * BOARD_SIZE

# vector to store vectors of 3D points for each checkerboard image
obj_points = []
# vector to store vectors of 2D points for each checkerboard image
img_points = []

images = []  # list for images
count = 0  # counter for images

# choosing some images for calibration
for path in os.listdir(DATA_DIR):
    cap = cv2.VideoCapture(os.path.join(DATA_DIR, path))
    while True:
        frame_grabbed, img = cap.read()
        if not frame_grabbed:
            break
        if count % 50 == 0:
            images.append(img)
        count += 1

print('Количество изображений для калибровки: {}'.format(len(images)))

for img in images:
    # converting the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # finding the chessboard corners
    # if desired number of corners are found in the image then ret == true
    ret, corners = cv2.findChessboardCorners(gray, BOARD_DIM, cv2.CALIB_CB_ADAPTIVE_THRESH
                                             + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        obj_points.append(obj_ps)

        # refining pixel coordinates for given 2d points
        img_ps = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(img_ps)

# camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, images[0].shape[:2], None, None)

# list for all camera positions
cam_positions = []

# TODO: do for all frames
for img in images:
    # converting the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # finding the chessboard corners
    # if desired number of corners are found in the image then ret == true
    ret, corners = cv2.findChessboardCorners(gray, BOARD_DIM,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        # refining pixel coordinates for given 2d points
        img_ps = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        ret, rvec, tvec = cv2.solvePnP(obj_ps, img_ps, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)

        # rotation matrix
        rot_mat = cv2.Rodrigues(rvec)[0]

        # camera position
        cam_pos = -np.matrix(rot_mat).T * np.matrix(tvec)

        cam_positions.append(cam_pos)

cam_positions = np.array(cam_positions)

# plotting the camera's movement
fig: plt.Figure = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2])
ax.plot_surface(np.arange(-100, 701, 100), np.arange(-100, 701, 100), np.zeros((9, 9)))
fig.show()
