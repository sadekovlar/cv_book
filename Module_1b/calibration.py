import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Defining the dimensions of checkerboard
CHECKERBOARD = (7, 7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []
# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None
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

for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret is True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    cv2.imshow('img', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
h, w = img.shape[:2]
"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)


image = images[0]
newCameraMatrix, validPixROI = cv2.getOptimalNewCameraMatrix(mtx, dist,(w, h), 1, (w, h))
undistorted_image = cv2.undistort(
    image, mtx, dist, None, newCameraMatrix
)
cv2.imshow("undistorted", undistorted_image)
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
intrinsic = mtx
rvec = rvecs[0]
tvec = tvecs[0]
# Project the points
image_points, jacobian = cv2.projectPoints(object_points, rvec, tvec, intrinsic, dist)
# Plot the points (using PyPlot)
plt.scatter(*zip(*image_points[:, 0, :]), marker='.')
plt.axis('equal')
plt.grid()
plt.show()