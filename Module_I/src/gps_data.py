from math import pi
import cv2 as cv
import pandas as pd
import os


class GpsData:
    def __init__(self):
        self.nordArr = []
        self.eastArr = []
        self.altArr = []
        self.speedArr = []
        self.yawArr = []
        self.grabMsec = []
        self.s = cv.FileStorage()

    def collect_data(self, filename):
        self.s.open(filename, cv.FILE_STORAGE_READ)
        shots = self.s.getNode("shots")
        for i in range(shots.size()):
            # print(shots.at(i).isSeq())
            # print(shots.at(i).isMap())
            if shots.at(i).isMap():
                mapping = shots.at(i)
                # print(mapping.keys())
                # print(dir(mapping))
                emlidLeft = mapping.getNode("emlidLeft")
                if emlidLeft.name() == "emlidLeft":
                    senseData = emlidLeft.getNode("senseData")
                    nord = senseData.getNode("nord")
                    east = senseData.getNode("east")
                    alt = senseData.getNode("alt")
                    speed = senseData.getNode("speed")
                    yaw = senseData.getNode("yaw")
                    grabMsec = mapping.getNode("grabMsec")
                    self.nordArr.append(nord.real())
                    self.eastArr.append(east.real())
                    self.altArr.append(alt.real())
                    self.speedArr.append(speed.real())
                    self.yawArr.append(yaw.real())
                    self.grabMsec.append(int(grabMsec.real()))
        self.s.release()
        self.save_to_csv()

    def save_to_csv(self):
        maindf = pd.DataFrame(data=None, columns=["nord", 'east', 'alt', 'speed', 'yaw', 'grabMsec'])
        for i in range(len(self.nordArr)):
            df = pd.DataFrame(data=[[self.nordArr[i], self.eastArr[i], self.altArr[i], self.speedArr[i], self.yawArr[i],
                                     self.grabMsec[i]]], columns=["nord", 'east', 'alt', 'speed', 'yaw', 'grabMsec'])
            maindf = pd.concat([maindf, df])
        if os.path.getsize("../data/tram/gps-data.csv") == 0:
            maindf.to_csv("../data/tram/gps-data.csv", sep=';', encoding='utf-8', index=False, mode='a', header=True)
        else:
            maindf.to_csv("../data/tram/gps-data.csv", sep=';', encoding='utf-8', index=False, mode='a', header=False)

    @staticmethod
    def read_from_csv(filename):
        result = pd.read_csv(filename, sep=';', index_col=False)
        return result

    @staticmethod
    def erase_content(filename):
        open(filename, "w").close()