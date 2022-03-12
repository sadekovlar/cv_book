from math import cos, sin
from pandas import array
import requests
import cv2

from season_reader import SeasonReader
from sence_data import SenseData

class Telemetry(SeasonReader):
    FUSTY_COUNT = 30 # после 30 кадров SenceData усторевает и уже не выводится 

    _currentSenceData: SenseData
    _actualityDataCount: int = 31 # Отслеживает, если данные устарели, то мы их не выводим  
    _addMapOnFrame: bool = True
    _addLogoOnFrame: bool = True

    #Предзагруженные изображения для отрисовки
    _logo_img: array = None
    _map_img: array = None

    def on_init(self):
        #Сразу загружаем logo
        self._logo_img = cv2.imread('./Module_I/img/logo_misis_en_small.jpg', cv2.IMREAD_COLOR)
        return True

    def upload_new_map(self):
        #https://yandex.ru/dev/maps/staticapi/doc/1.x/dg/concepts/input_params.html

        request = 'https://static-maps.yandex.ru/1.x/?ll='+str(self._currentSenceData._east)+','+str(self._currentSenceData._nord)+'&size=150,150&z=19&l=map'
        request = request + '&pt='+str(self._currentSenceData._east)+','+str(self._currentSenceData._nord)+',round'#add marker

        with open('./Module_I/img/map.jpg', 'wb') as handle:
            response = requests.get(request, stream=True)

            if not response.ok:
                print(response)

            for block in response.iter_content(1024):
                if not block:
                    break

                handle.write(block)
        self._map_img = cv2.imread('./Module_I/img/map.jpg', cv2.IMREAD_COLOR)
            
    def putMapOnFrame(self):         
        m_rows, m_cols, _ = self._map_img.shape
        rows,cols,_ = self.frame.shape
        roi = self.frame[rows-m_rows:rows, cols-m_cols:cols]

        dst = cv2.addWeighted(roi,1,self._map_img,0.5,0)
        self.frame[rows-m_rows:rows, cols-m_cols:cols] = dst

    def putSenceData(self):
        if self._actualityDataCount > self.FUSTY_COUNT:
            return 0
        
        #Вывод скорости
        SD = self._currentSenceData
        speed_mc = round(SD._speed, 3)
        speed_kh = SD.ms2kmh()

        cv2.putText(self.frame, f"Speed: {speed_kh} km/h ({speed_mc} m/s)", (15, 450),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        

        #Вывод координат
        geo = SD.convert_geo()#°
        nord_str = str(geo['nord']['grad']) + 'D' + str(geo['nord']['minute']) + "'"+ str(geo['nord']['sec']) +'\"'
        east_str = str(geo['east']['grad']) + 'D' + str(geo['east']['minute']) + "'"+ str(geo['east']['sec']) +'\"'
        cv2.putText(self.frame, f"Altitude: {SD._alt} m", (630, 520),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        cv2.putText(self.frame, f"Nord: {nord_str}", (630, 480),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        cv2.putText(self.frame, f"East: {east_str}", (630, 500),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)

        #Добавление карты
        if (self._addMapOnFrame):
            self.putMapOnFrame()

        
        #Вывод компасса
        R = 25
        x0, y0 = (230, 500)
        x1, y1 = int(x0 + sin(SD._yaw)*R), int(y0 - cos(SD._yaw)*R)
        cv2.arrowedLine(self.frame, (x0,y0), (x1,y1), (0, 255, 0), 2)
        cv2.putText(self.frame, f"N: {int(SD.rad2deg())}", (x1-55, y1+4),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        cv2.circle(self.frame, (x0,y0), R+3,(0, 0, 0), 1)

        #Вывод маркера времени получения SenceData
        # cv2.putText(self.frame, f"{sd._timestamp}", (15, 420),
        #         cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        pass
 
    def putMISISLogo(self):       
        l_rows, l_cols, _ = self._logo_img.shape
        rows,cols,_ = self.frame.shape
        roi = self.frame[rows-l_rows:rows, 0:l_cols:]

        dst = cv2.addWeighted(roi,1,self._logo_img,0.8,0)
        self.frame[rows-l_rows:rows, 0:l_cols] = dst
        pass

    def on_frame(self):
        if self._addLogoOnFrame:
            self.putMISISLogo()

        self.putSenceData()
        return True

    def on_gps_frame(self):       
        #Загрузка данных
        self._currentSenceData = SenseData(self.shot[self._gps_name])
        self._actualityDataCount = 1
        
        #Загрузка карты
        if (self._addMapOnFrame):
            self.upload_new_map()

        return True
    
    def on_imu_frame(self):
        return True

    def on_shot(self):
        return True

if __name__ == "__main__":

    for number in range(235, 236):
        init_args = {
            'path_to_data_root' : './data/tram/'
        }
        s = Telemetry()
        s.initialize(**init_args)
        s.run()
    print("Done!")

    
