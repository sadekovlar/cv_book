import cv2
import math as mt
import numpy as np
from season_reader import SeasonReader
from Module_I.load_calib import CalibReader

class MyReader(SeasonReader):
    def on_init(self):
        par = ["K", "D", "r", "t" ]
        calib = CalibReader()
        calib.initialize(file_name = './data/tram/leftImage.yml', param = par)
        self.matrix = calib.read()
        return True

    def on_shot(self):
        return True

    def on_frame(self):
        Xw =  np.array([15, 0, 0])
        r = self.matrix["r"].reshape(-1)
        t = self.matrix["t"].reshape(3, 1)
        K = self.matrix["K"]
        R = self.get_R(r)
        Xc = R.inv() @ Xw + t.T
        u = Xc[0,0]/Xc[0,1]*K[0,0] + K[0,2]
        v = Xc[0,2]/Xc[0,1]*K[1,1] + K[1,2]
        cv2.putText(self.frame, f"GrabMsec: {self.frame_grab_msec}", (15, 50),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        return True

    def get_R(self, r):
        cos_x = mt.cos(r[0])
        sin_x = mt.sin(r[0])
        cos_y = mt.cos(r[1])
        sin_y = mt.sin(r[1])
        cos_z = mt.cos(r[2])
        sin_z = mt.sin(r[2])
        R_x = np.array([[1, 0, 0],          [0, cos_x, -sin_x], [0, sin_x,  cos_x]])
        R_y = np.array([[cos_y, 0, sin_y],  [0, 1, 0],          [-sin_y, 0, cos_y]])
        R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0],  [0, 0,  1]])
        return R_z * R_y * R_x


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

    
