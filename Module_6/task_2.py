import numpy as np
import cv2
import time
from srccam.season_reader import SeasonReader


class MovingObjects(SeasonReader):

    def on_init(self):
        self.lk_params = dict(winSize=(40, 40),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners =20,
                      qualityLevel = 0.3,
                      minDistance = 10,
                      blockSize = 7)
        self.trajectory_len = 20
        self.detect_interval = 1
        self.trajectories = []
        self.frame_idx = 0
        self.prev_gray = None
        return True

    def track_moving(self, frame):
        start = time.time()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = frame.copy()
        # prev_gray = frame_gray

        if len(self.trajectories) > 0:
            img0, img1 = self.prev_gray, frame_gray
            p0 = np.float32([trajectory[-1] for trajectory in self.trajectories]).reshape(-1, 1, 2)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
            p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
            good = d < 1

            new_trajectories = []
            for trajectory, (x, y), good_flag in zip(self.trajectories, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                trajectory.append((x, y))
                if len(trajectory) > self.trajectory_len:
                    del trajectory[0]
                new_trajectories.append(trajectory)
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

            self.trajectories = new_trajectories

            cv2.polylines(img, [np.int32(trajectory) for trajectory in self.trajectories], False, (0, 255, 0))
            cv2.putText(img, "track count : %d" % len(self.trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 25, 162), 4)

        if self.frame_idx % self.detect_interval == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255

            for x, y in [np.int32(trajectory[-1]) for trajectory in self.trajectories]:
                cv2.circle(mask, (x, y), 5, 0, -1)

            p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    self.trajectories.append([(x, y)])

        self.frame_idx += 1
        self.prev_gray = frame_gray

        end = time.time()
        fps = 1 / (end - start)

        cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 25, 162), 4)
        cv2.imshow("Optical flow", img)
        cv2.imshow("Mask", mask)

    def on_shot(self):
        return True

    def on_frame(self):
        self.track_moving(self.frame)
        return True

    def on_gps_frame(self) -> bool:
        return True


if __name__ == "__main__":
    for number in range(235, 236):
        init_args = {
            'path_to_data_root': '../data/city/'
        }
        s = MovingObjects()
        s.initialize(**init_args)
        s.run()
    print("Done!")
