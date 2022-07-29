import cv2
import numpy as np
from skimage import measure


class BacklightReducer:
    # Класс для уменьшения засветки вызванной снежной бурей

    @staticmethod
    def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
        look_up_table = np.empty((1, 256), np.uint8)
        for i in range(256):
            look_up_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        res = cv2.LUT(img, look_up_table)
        return res

    @staticmethod
    def get_map_mask(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        _, thresh_img = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)
        thresh_img = cv2.erode(thresh_img, None, iterations=2)
        thresh_img = cv2.dilate(thresh_img, None, iterations=4)
        # perform a connected component analysis on the thresholded image,
        # then initialize a mask to store only the "large" components
        labels = measure.label(thresh_img, background=0)
        mask = np.zeros(thresh_img.shape, dtype="uint8")
        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue
            # otherwise, construct the label mask and count the
            # number of pixels
            labelMask = np.zeros(thresh_img.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            # if the number of pixels in the component is sufficiently
            # large, then add it to our mask of "large blobs"
            if numPixels > 300:
                mask = cv2.add(mask, labelMask)
        return mask

    @staticmethod
    def reduce_lights_struck(img: np.ndarray) -> np.ndarray:
        res = img.copy()
        overexposed_mask = BacklightReducer.get_map_mask(res)
        mask_inv = cv2.bitwise_not(overexposed_mask)
        non_overexposed_part = cv2.bitwise_and(res, res, mask=mask_inv)
        overexposed_part = cv2.bitwise_and(res, res, mask=overexposed_mask)
        corrected = BacklightReducer.gamma_correction(overexposed_part, 1.06)
        res = cv2.add(non_overexposed_part, corrected)
        return res


if __name__ == '__main__':
    video_name = 'klt.544.006.mp4'
    cap = cv2.VideoCapture('../data/processing/' + video_name)
    while cap.isOpened():
        succeed, frame = cap.read()
        if succeed:
            original = frame.copy()
            mask = BacklightReducer.get_map_mask(original)
            frame = BacklightReducer.reduce_lights_struck(frame)
            # cv2.imshow("mask", mask)
            # cv2.imshow("original", original)
            cv2.imshow(video_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cv2.destroyAllWindows()
            cap.release()
        cv2.waitKey(1)
