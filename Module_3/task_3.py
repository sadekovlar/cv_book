import string
from tokenize import String
import cv2 as cv
import numpy as np

class Reader:
    """Обработка видеопотока."""
    def initialize(self, path_to_video):
        self.videos = [path_to_video,] if isinstance(path_to_video, str) else path_to_video

    def nextclip(self):
        if len(self.videos) > 0:
            print("Run video: ", self.videos[0])
            self.cap = cv.VideoCapture(self.videos[0])
            del self.videos[0]
            return True
        return False

    def run(self):
        while self.nextclip():
            if not self.cap.isOpened():
                print("Ошибка! Не удалось открыть файл.")
            else:
                while self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret: break
                    self.on_frame(frame)
                    cv.imshow('VideoPlayer', frame)
                    if cv.waitKey(10) & 0xFF == ord('q'):
                        break
                self.cap.release()
        cv.destroyAllWindows()

    def on_frame(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Harris
        harris = cv.cornerHarris(gray, 2, 3, 0.04)
        harris = cv.dilate(harris, (10,10))

        # Fast
        fast = cv.FastFeatureDetector_create()
        fast.setThreshold(90)
        kp = fast.detect(frame, None)

        # Drawing
        hPoints = np.argwhere(harris>0.25*harris.max())
        for h in hPoints:
            cv.circle(frame, (h[1],h[0]), 5, (0,0,255), 1, 2)                     # Harris / RED
        cv.drawKeypoints(frame, kp, outImage=frame, color=(255,0,0))              # Fast / BLUE

        # Near points:
        fPoints = cv.KeyPoint_convert(kp)

        countEquals = 0
        for h in hPoints:
            if h.T in fPoints:
                cv.circle(frame, (h[1],h[0]), 2, (0,255,128), 1, 1)               # Совпадения / GREEN
                countEquals += 1
        total = len(hPoints) + len(fPoints)

        cv.putText(frame, 'Harris: ' + str(len(hPoints)), (10,430), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255), 1, cv.LINE_AA)
        cv.putText(frame, 'FAST:   ' + str(len(fPoints)), (10,450), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255,0,0), 1, cv.LINE_AA)
        cv.putText(frame, 'All:     ' + str(total), (10,470), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200,200,200), 1, cv.LINE_AA)
        cv.putText(frame, 'Equals: ' + str(countEquals), (10,500), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (150,250,150), 1, cv.LINE_AA)

        cv.putText(frame, str(round(len(hPoints)/total*100, 2)) + "%", (150,430), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255), 1, cv.LINE_AA)
        cv.putText(frame, str(round(len(fPoints)/total*100, 2)) + "%", (150,450), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255,0,0), 1, cv.LINE_AA)
        cv.putText(frame, str(round(countEquals/total*200, 2)) + "%", (150,500), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (150,250,150), 1, cv.LINE_AA)


if __name__ == '__main__':
    init_args = {
        'path_to_video': [
            '../data/city/trm.169.007.avi',
            '../data/tram/trm.169.008.avi'
        ]
    }
    s = Reader()
    s.initialize(**init_args)
    s.run()
    print('Done!')