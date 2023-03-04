import os

import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

print('Number of images for calibration: {}'.format(len(images)))


def change_order(points: np.ndarray) -> np.ndarray:
    """Function for changing the order of detected chessboard corners."""

    # coordinates of the 4 outermost corners
    xs = sorted([points[0][0, 0], points[BOARD_DIM[0] - 1][0, 0], points[-BOARD_DIM[0]][0, 0], points[-1][0, 0]])
    ys = sorted([points[0][0, 1], points[BOARD_DIM[0] - 1][0, 1], points[-BOARD_DIM[0]][0, 1], points[-1][0, 1]])

    if xs.index(points[0][0, 0]) < 2 and ys.index(points[0][0, 1]) < 2:
        # if no changes are required
        return points

    elif xs.index(points[BOARD_DIM[0] - 1][0, 0]) < 2 and ys.index(points[BOARD_DIM[0] - 1][0, 1]) < 2:
        # if points are rotated clockwise
        return np.rot90(points.reshape(BOARD_DIM[::-1] + (1, 2))).reshape((BOARD_DIM[0] * BOARD_DIM[1], 1, 2))

    elif xs.index(points[-BOARD_DIM[0]][0, 0]) < 2 and ys.index(points[-BOARD_DIM[0]][0, 1]) < 2:
        # if points are rotated counterclockwise
        return np.rot90(points.reshape(BOARD_DIM[::-1] + (1, 2)), -1).reshape((BOARD_DIM[0] * BOARD_DIM[1], 1, 2))

    # if points are reversed
    return points[::-1, ...]


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
        img_ps = cv2.cornerSubPix(gray, np.ascontiguousarray(change_order(corners)), (11, 11), (-1, -1), criteria)
        img_points.append(img_ps)

# camera calibration
print('Calibration in progress...')
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, images[0].shape[:2], None, None)
print('Calibration complete.')

# number of previous camera positions to plot
PATH_LEN = 50

# list for all camera positions
cam_positions = [[], [], []]

# creating the graph for the camera position
fig: plt.Figure = plt.figure(figsize=(3.4, 3.0))
ax: Axes3D = fig.add_subplot(projection='3d')

# prepare the data for rendering the chessboard
x = np.arange(-100, 701, 100)
y = np.arange(-100, 701, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros((9, 9))

for path in os.listdir(DATA_DIR):
    cap = cv2.VideoCapture(os.path.join(DATA_DIR, path))
    while True:
        frame_grabbed, img = cap.read()
        if not frame_grabbed:
            break

        # converting the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # finding the chessboard corners
        # if desired number of corners are found in the image then ret == true
        ret, corners = cv2.findChessboardCorners(gray, BOARD_DIM, cv2.CALIB_CB_ADAPTIVE_THRESH
                                                 + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            # for some reason, findChessboardCorners sometimes returns points in the wrong order
            # this function I wrote fixes the order if necessary
            corners = np.ascontiguousarray(change_order(corners))

            # refining pixel coordinates for given 2d points
            img_ps = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # calculating the pose of the chessboard
            _, rvec, tvec = cv2.solvePnP(obj_ps, img_ps, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)

            # drawing the chessboard corners
            img = cv2.drawChessboardCorners(img, BOARD_DIM, img_ps, ret)

            # rotation matrix
            rot_mat = cv2.Rodrigues(rvec)[0]

            # camera position
            cam_pos = -np.matrix(rot_mat).T * np.matrix(tvec)

            # adding the current position to the path
            for axis in range(3):
                cam_positions[axis].append(cam_pos[axis, 0])

            # deleting the oldest point if necessary
            if len(cam_positions[0]) > PATH_LEN:
                for axis in range(3):
                    cam_positions[axis] = cam_positions[axis][1:]

            # drawing the chessboard
            ax.plot_surface(X, Y, Z, color='green')
            # drawing the camera's path
            ax.plot(cam_positions[0], cam_positions[1], cam_positions[2], color='blue', linewidth=2)
            # drawing the current position of the camera
            ax.scatter(cam_pos[0, 0], cam_pos[1, 0], cam_pos[2, 0], color='red', linewidths=3)

            # making the graph have the same scale for all axes
            ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

            ax.invert_zaxis()  # inverting the z-axis

            fig.canvas.draw()  # drawing the plot
            w, h = fig.canvas.get_width_height()  # getting dimensions of the plot
            # converting the graph image into a numpy array
            plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))

            # overlaying the graph on the bottom left or the bottom right corners the frame
            if img_ps[0][0, 0] < img.shape[1] // 2:
                img[-plot_img.shape[0]:, -plot_img.shape[1]:, ::-1] = plot_img
            else:
                img[-plot_img.shape[0]:, :plot_img.shape[1], ::-1] = plot_img

            ax.clear()  # clearing the graph

        cv2.imshow('camera motion', img)
        cv2.waitKey(10)

    cap.release()
