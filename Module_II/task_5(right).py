#Уменьшить засветку от света фар встречного автомобиля [klt.427.003]

import math

import cv2



class Better:
    @staticmethod
    def bt(self):
        a = cv2.GaussianBlur(self, (11, 11), cv2.BORDER_DEFAULT)
        a = cv2.addWeighted(self, 1.5, a, -1.0, 0, 1)
        return a


if __name__ == '__main__':
    video_name = 'trm.168.091.avi'
    cap = cv2.VideoCapture('../data/processing/' + video_name)
    while cap.isOpened():
        succeed, frame = cap.read()
        if succeed:
            frame = Better.bt(frame)
            cv2.imshow(video_name, frame)
        else:
            cv2.destroyAllWindows()
            cap.release()
        cv2.waitKey(1)
