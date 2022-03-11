import cv2

from season_reader import SeasonReader

from Module_I.example_4_trajectory.trajectory_estimator import TrajectoryEstimator
from Module_I.load_calib import CalibReader


class ReaderForTrajectory(SeasonReader):
    def on_init(self):
        par = ['K', 'D', 'r', 't']
        calib_reader = CalibReader(
            file_name=r'../../data/tram/leftImage.yml',
            param=par)
        calib_dict = calib_reader.read()
        self.traj_estimator = TrajectoryEstimator(calib_dict=calib_dict,
                                                  height=1.6,
                                                  wight=1.6,
                                                  length=25,
                                                  depth=8)

        return True

    def on_shot(self):
        return True

    def on_frame(self):
        cv2.putText(self.frame, f'GrabMsec: {self.frame_grab_msec}', (15, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        self.traj_estimator.dray_trajectory(self.frame)

        return True

    def on_gps_frame(self):
        shot: dict = self.shot[self._gps_name]['senseData']
        shot['grabMsec'] = self.shot[self._gps_name]['grabMsec']
        return True

    def on_imu_frame(self):
        shot: dict = self.shot[self._imu_name]
        return True


if __name__ == "__main__":
    init_args = {
        'path_to_data_root': '../../data/tram/'
    }
    s = ReaderForTrajectory()
    s.initialize(**init_args)
    s.run()
    print('Done!')

