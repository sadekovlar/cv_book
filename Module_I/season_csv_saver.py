import cv2
import pandas as pd
from season_reader import SeasonReader

class MyReader(SeasonReader):
    def on_init(self):
        self.gps_data = list()
        self.imu_data = list()
        self.can_data = list()
        return True

    def on_shot(self):
        gps_frame: pd.DataFrame = self.get_gps_frame()
        imu_frame: pd.DataFrame = self.get_imu_frame()
        can_frame: pd.DataFrame = self.get_can_frame()
        if len(gps_frame) > 0:
            self.gps_data.append(gps_frame)
        if len(imu_frame) > 0:
            self.imu_data.append(imu_frame)
        if len(can_frame) > 0:
            self.can_data.append(can_frame)    
        return True

if __name__ == "__main__":

    for number in range(235, 236):
        init_args = {
            "serial": "trm",
            "season": number,
            "video_ext": "avi",
            "start_episode": 1,
            "finish_episode": 999,
            'gps_name': 'emlidLeft',
            'camera_name': 'central60',
            'can_name': 'dbwFbTram'
        }
        s = MyReader()
        s.initialize(**init_args)
        s.run()
        name = s._data_path.replace("/",".").replace("\\",".")
        if len(s.gps_data):
            pd.DataFrame(s.gps_data).to_csv("testdata/gps"+ name +".csv", index=False)
        if len(s.imu_data):
            pd.DataFrame(s.imu_data).to_csv("testdata/imu"+ name +".csv", index=False)
        if len(s.can_data):
            pd.DataFrame(s.can_data).to_csv("testdata/can"+ name +".csv", index=False)
        print('done!')
    print("Done!")


    """
    klt
    init_args = {
        "path_to_data_root" : "С:\\testdata",
        "serial": "klt",
        "season": 574,
        "video_ext": "mp4",
        "start_episode": 1,
        "finish_episode": 999,
        'gps_name': 'ubloxGps',
        'camera_name': 'leftImage',
        'imu_name': 'minsEth',
        'can_name': 'dbwFbVehicleCan'
    }
    """