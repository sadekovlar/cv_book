import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Extracting path of individual image stored in a given directory
main_path = "../data/calib"
images = list()
fs = os.listdir(main_path)
count = 0
for path in fs:
    cap = cv2.VideoCapture(os.path.join(main_path, path))
    f = True
    while f:
        f, im = cap.read()
        if count % 50 == 0:
            images.append(im)
        count = count + 1

CHECKERBOARD = (7, 7)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
# 3d point in real world space
objpoints = []
# 2d points in image plane
imgpoints = []
for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret is True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

_img_shape = img[:2]
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")


h, w = img.shape[:2]
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
v = np.concatenate((undistorted_img, img), axis=1)
cv2.imshow("undistorted", v)
cv2.waitKey(0)
cv2.destroyAllWindows()


# The distortion matrix that I vary
# Generate Grid of Object Points
grid_size, square_size = [20, 20], 0.2
object_points = np.zeros([grid_size[0] * grid_size[1], 3])
mx, my = [(grid_size[0] - 1) * square_size / 2, (grid_size[1] - 1) * square_size / 2]
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        object_points[i * grid_size[0] + j] = [i * square_size - mx, j * square_size - my, 0]

# Setup the camera information
rvec = rvecs[0]
tvec = tvecs[0]
# Project the points
image_points, jacobian = cv2.projectPoints(object_points, rvec, tvec, K, D)
# Plot the points (using PyPlot)
plt.scatter(*zip(*image_points[:, 0, :]), marker='.')
plt.axis('equal')
plt.grid()
plt.show()