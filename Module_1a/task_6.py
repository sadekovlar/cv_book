from math import cos, sin
from pandas import array
import requests
import cv2

from srccam.season_reader import SeasonReader
from srccam.sense_data import SenseData

DISPLAY_SPEC = {
    'font': cv2.FONT_HERSHEY_PLAIN,
    'scale': 1.0,
    'color': (0, 255, 0),  # Green
    'thickness': 2
}


# Добавление на кадр информации об телеметрии
class Telemetry(SeasonReader):
    FUSTY_COUNT = 30  # после 30 кадров SenseData устаревает и уже не выводится
    _currentSenseData: SenseData
    _actualityDataCount: int = 31  # Отслеживает, если данные устарели, то мы их не выводим
    _addMapOnFrame: bool = True
    _addLogoOnFrame: bool = True
    # Предзагруженные изображения для отрисовки
    _logo_img: array = None
    _map_img: array = None

    def on_init(self, _addMapOnFrame=True, _addLogoOnFrame=True):
        # Сразу загружаем logo
        self._logo_img = cv2.imread('./Module_1a/img/logo_misis_en_small.jpg', cv2.IMREAD_COLOR)
        self._addMapOnFrame = _addMapOnFrame
        self._addLogoOnFrame = _addLogoOnFrame
        return True

    def upload_new_map(self):
        # https://yandex.ru/dev/maps/staticapi/doc/1.x/dg/concepts/input_params.html
        request = 'https://static-maps.yandex.ru/1.x/?ll=' + str(self._currentSenseData._east) + ',' + str(
            self._currentSenseData._nord) + '&size=150,150&z=19&l=map'
        request = request + '&pt=' + str(self._currentSenseData._east) + ',' + str(
            self._currentSenseData._nord) + ',round'  # add marker

        with open('./Module_1a/img/map.jpg', 'wb') as handle:
            response = requests.get(request, stream=True)
            if not response.ok:
                print(response)
            for block in response.iter_content(1024):
                if not block:
                    break
                handle.write(block)
        self._map_img = cv2.imread('./Module_1a/img/map.jpg', cv2.IMREAD_COLOR)

    def put_map_on_frame(self):
        m_rows, m_cols, _ = self._map_img.shape
        rows, cols, _ = self.frame.shape
        roi = self.frame[rows - m_rows:rows, cols - m_cols:cols]
        dst = cv2.addWeighted(roi, 1, self._map_img, 0.5, 0)
        self.frame[rows - m_rows:rows, cols - m_cols:cols] = dst

    def put_sense_data(self):
        if self._actualityDataCount > self.FUSTY_COUNT:
            return 0
        # Вывод скорости
        sd = self._currentSenseData
        speed_mc = round(sd._speed, 3)
        speed_kh = sd.ms2kmh()
        cv2.putText(self.frame, f"Speed: {speed_kh} km/h ({speed_mc} m/s)", (15, 450),
                    DISPLAY_SPEC['font'], DISPLAY_SPEC['scale'], DISPLAY_SPEC['color'], DISPLAY_SPEC['thickness'])
        # Вывод координат
        nord_str, east_str = sd.get_geo_str()
        cv2.putText(self.frame, f"Altitude: {sd._alt} m", (630, 520),
                    DISPLAY_SPEC['font'], DISPLAY_SPEC['scale'], DISPLAY_SPEC['color'], DISPLAY_SPEC['thickness'])
        cv2.putText(self.frame, f"Nord: {nord_str}", (630, 480),
                    DISPLAY_SPEC['font'], DISPLAY_SPEC['scale'], DISPLAY_SPEC['color'], DISPLAY_SPEC['thickness'])
        cv2.putText(self.frame, f"East: {east_str}", (630, 500),
                    DISPLAY_SPEC['font'], DISPLAY_SPEC['scale'], DISPLAY_SPEC['color'], DISPLAY_SPEC['thickness'])
        # Добавление карты
        if self._addMapOnFrame:
            self.put_map_on_frame()

        # Вывод компаса
        # радиус компаса в пикселях
        r = 25
        # положение в кадре
        x0, y0 = (230, 500)
        x1, y1 = int(x0 + sin(sd._yaw) * r), int(y0 - cos(sd._yaw) * r)
        cv2.arrowedLine(self.frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(self.frame, f"N: {int(sd.rad2deg())}", (x1 - 55, y1 + 4),
                    DISPLAY_SPEC['font'], DISPLAY_SPEC['scale'], DISPLAY_SPEC['color'], DISPLAY_SPEC['thickness'])
        cv2.circle(self.frame, (x0, y0), r + 3, (0, 0, 0), 1)
        # Вывод маркера времени получения SenseData
        # cv2.putText(self.frame, f"{sd._timestamp}", (15, 420),
        #           DISPLAY_SPEC['font'], DISPLAY_SPEC['scale'],DISPLAY_SPEC['color'],DISPLAY_SPEC['thickness'])
        pass

    def put_misis_logo(self):
        l_rows, l_cols, _ = self._logo_img.shape
        rows, _, _ = self.frame.shape
        roi = self.frame[rows - l_rows:rows, 0:l_cols:]
        dst = cv2.addWeighted(roi, 1, self._logo_img, 0.8, 0)
        self.frame[rows - l_rows:rows, 0:l_cols] = dst
        pass

    def on_frame(self):
        if self._addLogoOnFrame:
            self.put_misis_logo()
        self.put_sense_data()
        return True

    def on_gps_frame(self):
        # Загрузка данных
        self._currentSenseData = SenseData(self.shot[self._gps_name])
        self._actualityDataCount = 1
        # Загрузка карты
        if self._addMapOnFrame:
            self.upload_new_map()

        return True

    def on_imu_frame(self):
        return True

    def on_shot(self):
        return True


if __name__ == "__main__":
    init_args = {
        'path_to_data_root': '../data/city/'
    }
    s = Telemetry()
    s.initialize(**init_args)
    s.run()
    print("Done!")
