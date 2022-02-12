import cv2
from season_reader import SeasonReader

class MyReader(SeasonReader):
    def on_init(self):
        return True

    def on_shot(self):
        return True

    def on_frame(self):
        cv2.putText(self.frame, f"GrabMsec: {self.frame_grab_msec}", (15, 50),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        return True

    def on_gps_frame(self):
        shot: dict = self.shot[self._gps_name]['senseData']
        shot['grabMsec'] = self.shot[self._gps_name]['grabMsec']
        return True

        
    def on_imu_frame(self):
        shot: dict = self.shot[self._imu_name]
        return True

if __name__ == "__main__":

    for number in range(235, 236):
        init_args = {
            'path_to_data_root' : './data/tram/'
        }
        s = MyReader()
        s.initialize(**init_args)
        s.run()
    print("Done!")

    
