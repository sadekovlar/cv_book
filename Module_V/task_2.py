import math
import cv2 as cv
import numpy as np
from Module_I.load_calib import CalibReader
from Module_I.season_reader import SeasonReader
from Module_I.spatial_geometry_tools.calib import Calib
from Module_I.spatial_geometry_tools.camera import Camera
from Module_I.spatial_geometry_tools.point import Point3d as Point

class RailsDrawer:

    def __init__(self, calib_dict):
        self.calib = Calib(calib_dict)
        self.camera = Camera(self.calib)
        self.A_inv = self.count_matrix_for_2d_3d_projection()
        self.offset = 0 # 2d offset
        self.l, self.r, self.dv = -0.8, 0.8, 0.15

    def rp_23d(self, point2d: None): # project 2d point to 3d point
        if point2d is None:
            point2d = []
        h = 0  # Процекция по земле, следовательно высота нулевая
        p_ = self.A_inv @ Point((point2d[0], point2d[1], 1)).vec
        error = abs(self.calib.t)/2 # Погрешность перевода между С.К.
        x = p_[0] / p_[2] + error[0]
        y = p_[1] / p_[2] + error[1]
        z = h
        return Point((x, y, z))

    def count_matrix_for_2d_3d_projection(self):
        R = self.calib.cam_to_vr @ self.calib.r
        affine_matrix = np.concatenate((R, -R @ self.calib.t), 1)
        P = self.calib.K @ affine_matrix
        A = self.get_A_from_P_on_floor(P)
        A_inv = np.linalg.inv(A)
        return A_inv

    def get_A_from_P_on_floor(self, P: np.ndarray) -> np.ndarray:
        h = 0
        A = np.zeros((3, 3))
        A[0, 0], A[0, 1], A[0, 2] = P[0, 0], P[0, 1], h * P[0, 2] + P[0, 3]
        A[1, 0], A[1, 1], A[1, 2] = P[1, 0], P[1, 1], h * P[1, 2] + P[1, 3]
        A[2, 0], A[2, 1], A[2, 2] = P[2, 0], P[2, 1], h * P[2, 2] + P[2, 3]
        return A

    def getInterest(self, frame):
        w, h = frame.shape[1], frame.shape[0]
        ipoints = (
            (w//3.52+self.offset, h),
            (w//2.22+self.offset, h//2),
            (w//1.75+self.offset, h//2),
            (w//1.46+self.offset, h)
        )
        mask = np.zeros_like(frame)
        channel_count = frame.shape[2]
        match_mask_color = (255,) * channel_count
        cv.fillPoly(mask, np.array([ipoints], np.int32), match_mask_color)
        masked_frame = cv.bitwise_and(frame, mask)
        return (masked_frame, ipoints, (w, h))

    def mean(self, lst):
        return sum(lst)/len(lst)

    def draw_rails(self, frame):
        mask, borders, size = self.getInterest(frame)
        gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        canny = cv.Canny(gray, 100, 250, 3)
        cv.imshow('111', canny)
        linesP = cv.HoughLinesP(canny, 1, np.pi / 180, 50, None, 50, 3)
        
        l, r = [], []
        if linesP is not None:
            for i in range(0, len(linesP)):
                ls = linesP[i][0]

                # 2d points:
                a, b = (ls[0],ls[1]), (ls[2],ls[3])
                if ls[0] > ls[2]: a, b = b, a
                
                # 3d points:
                a3 = self.rp_23d(a)
                b3 = self.rp_23d(b)
                n = math.sqrt((a3.x-b3.x)**2) # Расст. между точками прямой по горизонтали
                
                if n < 0.35:
                    if (self.l-self.dv) < a3.x < (self.l+self.dv):
                        l.append(a3.x)
                    elif (self.r-self.dv) < a3.x < (self.r+self.dv):
                        r.append(a3.x)
                    cv.line(frame, a, b, (0,0,255) if a[0] < self.offset + size[0]//2 else (255,0,255), 1, cv.LINE_AA) # all lines

        if len(l) > 0 and len(r) > 0:
            self.l, self.r  = self.mean(l), self.mean(r)
        elif len(l) > 0:
            self.l = self.mean(l)
            self.r = self.l + 1.52
        elif len(r) > 0:
            self.r = self.mean(r)
            self.l = self.r - 1.52
            #print(self.r - self.l)
        

        le, ri = self.camera.project_point_3d_to_2d(Point((self.l,4,0))), self.camera.project_point_3d_to_2d(Point((self.r,4,0)))
        self.offset = (le[0]+ri[0])//2 - size[0]//2

        cv.circle(frame, le, 2, (255,0,0), 2)
        cv.circle(frame, ri, 2, (255,0,0), 2)
        cv.circle(frame, (self.offset + size[0]//2, 480), 5, (0,255,0), 3)


class Reader(SeasonReader):
    """Обработка видеопотока."""

    def on_init(self, _file_name: str = None):
        par = ['K', 'D', 'r', 't']
        calib_reader = CalibReader()
        calib_reader.initialize(
            file_name='../data/tram/leftImage.yml',
            param=par)
        calib_dict = calib_reader.read()
        self.rd = RailsDrawer(calib_dict)
        return True

    def on_shot(self):
        return True

    def on_frame(self):
        cv.putText(self.frame, f'GrabMsec: {self.frame_grab_msec}', (15, 50),
                    cv.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        self.rd.draw_rails(self.frame)
        return True

    def on_gps_frame(self):
        shot: dict = self.shot[self._gps_name]['senseData']
        shot['grabMsec'] = self.shot[self._gps_name]['grabMsec']
        return True

    def on_imu_frame(self):
        return True


if __name__ == '__main__':
    init_args = {
        'path_to_data_root': '../data/tram/'
    }
    s = Reader()
    s.initialize(**init_args)
    s.run()
    print('Done!')