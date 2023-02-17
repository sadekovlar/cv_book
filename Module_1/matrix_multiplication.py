import cv2
import numpy as  np

image = cv2.imread("image.jpg").astype(np.float32) / 255

permutation = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
], dtype=np.float32)

changed = image.dot(permutation)
cv2.imshow('permuted', changed)
cv2.waitKey(0)
cv2.destroyAllWindows()

changed = changed.dot(permutation)
cv2.imshow('permuted yet again', changed)
cv2.waitKey(0)
cv2.destroyAllWindows()

changed = changed.dot(permutation)
cv2.imshow('permuted to the initial state', changed)
cv2.waitKey(0)
cv2.destroyAllWindows()


more_red = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 3]
    ], dtype=np.float32)


cv2.imshow('more red', image.dot(more_red))
cv2.waitKey(0)
cv2.destroyAllWindows()


color_blind = np.array([
    [0.3, 0.3, 0.3],
    [0.3, 0.3, 0.3],
    [0.3, 0.3, 0.3]
])

cv2.imshow('color blind', image.dot(color_blind))
cv2.waitKey(0)
cv2.destroyAllWindows()