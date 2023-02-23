import cv2
import numpy as np

# Code assumes that the video lives in the following
# folder and is titled video.wmv.
folder_template = '../data/optical_flow/{0}'
# Reset frequency determines how many frames the
# stabilization will track before resetting to the
# identity transform. May switch to using a maximum
# translation, rotation, scale, etc.
reset_frequency = 100 

# Step 1 - Get frame-to-frame transform matrices
cap = cv2.VideoCapture(folder_template.format('nhd.002.001.left.avi'))
frames = [cap.read()[1][:, :, 0]]
transforms = [np.identity(3)]
height, width = frames[0].shape
while(cap.isOpened()):
    try:
        current = cap.read()[1][:, :, 0]
    except TypeError:
        break
    prev_corner = cv2.goodFeaturesToTrack(frames[-1], 200, 0.0001, 10);
    cur_corner, status, _ = cv2.calcOpticalFlowPyrLK(frames[-1], current, prev_corner, np.array([]))
    prev_corner, cur_corner = map(lambda corners: corners[status.ravel().astype(bool)], [prev_corner, cur_corner])
    transform, _ = cv2.estimateAffine2D(prev_corner, cur_corner, True)
    if transform is not None:
        transform = np.append(transform, [[0, 0, 1]], axis=0)
    if transform is None:
        transform = transforms[-1]
    transforms.append(transform)
    frames.append(current)
cap.release()

# Step #2 use the transforms to stabilize images
height, width = frames[0].shape
stabilized_frames = []
last_transform = np.identity(3)
for frame, transform, index in zip(frames, transforms, range(len(frames))):
    transform = transform.dot(last_transform)
    if index % reset_frequency == 0:
        transform = np.identity(3)
    last_transform = transform
    inverse_transform = cv2.invertAffineTransform(transform[:2])
    stabilized_frames.append(cv2.warpAffine(frame, inverse_transform, (width, height)))

writer = cv2.VideoWriter(folder_template.format('output.mp4'), 
                         cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
                         20.0, (width*2, height), False)
for frame, stabilized in zip(frames, stabilized_frames):
    writer.write(np.concatenate([frame, stabilized], axis=1))
writer.release()
print("done!")