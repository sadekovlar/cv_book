import os
import numpy as np
import cv2
import sys

from srccam.load_calib import CalibReader
import math 
import matplotlib.pyplot as plt


class VisualOdometry():
    def __init__(self, data_dir):
        par = ["K", "D", "r", "t"]
        calib = CalibReader()
        calib.initialize(file_name=f"{data_dir}/leftImage.yml",
                          param=par)
        matrix = calib.read()
        #self.gps = PathCreator(matrix.get('K'))
        #self.data_gps = self.gps.get_gps_data()
        #self.dists = self.gps.dists
        #self.angles = self.gps.angles
        self.K = matrix.get('K')
        self.P = np.pad(self.K, ((0,0),(0,1)), mode='constant', constant_values=0)

        self.images = self._load_images(data_dir)
        self.gt_poses = self._load_poses(self, len_images=len(self.images))
        self.orb = cv2.ORB_create(100000) 
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    @staticmethod
    def _image_count(filepath):
        '''
        Считаем количество изображений для поз
        '''
        img_folder = os.listdir(os.path.join(filepath,"image_l"))
        img_count = len(img_folder)
        return img_count

    @staticmethod
    def _load_poses(self, len_images):
        """
        Функция создает начальную позицию
        Параметры:
        len_images - количество фремов в видео
        Возвращает:
        poses (ndarray):  массив с позицией
        """

        poses = []
        for i in range(len_images):    #создаем позы по кол-ву изображений
            a = '1 0 -0 -0 -0 1 -0 -0 0 0 1 0'
            T = np.fromstring(a, dtype=np.float64, sep=' ')      #размерность [0:12]
            T = T.reshape(3, 4)     #размерность [3:4]
            T = np.vstack((T, [0, 0, 0, 1]))    #добавляет массив [0,0,0,1] в конец массива
            poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        """
        Функция загружает фото
        Параметры:
        filepath - путь к каталогу с видео (str)
        Возвращает:
        images (list): изображения в оттенках серого
        Описание:
        1)в лист image_paths по пути filepath добавляется сортированный список , содержащий имена файлов и директорий в каталоге
        2)изображения из листа image_paths преобразуются в массив ndarray с оттенком серого и добавляются в лист
        """
        video_name_list = os.listdir(filepath)
        video_name_list = [video for video in video_name_list if video.find(".avi") > 0]
        images = list()
        i = 0 # текущий кадр
        for name in video_name_list:
            path_ = os.path.join(filepath, name)
            cap = cv2.VideoCapture(path_)
            succeed = True
            while succeed:
                succeed, frame = cap.read()
                if succeed is False:
                    break
                i+=1
                if i%3 == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    images.append(frame)
            cap.release()       
        return images

    @staticmethod
    def _form_transf(R, t):
        """
        Функция создает матрицу преобразования из заданной матрицы поворота и вектора перемещения
        Параметры:
        R (ndarray): Матрица вращения
        t (list): Вектор перевода
        Возвращает:
        T (ndarray): Матрица преобразования
        """
        # двумерный массив у которого все элементы по диагонали равны 1, а все остальные равны 0
        T = np.eye(4, dtype=np.float64) # 4 -количество строк выходного массива
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        """
        Эта функция обнаруживает и вычисляет ключевые точки, дескриптор из i-1-го и i изображений, используя класс или объект
        Параметры:
        i - Текущий кадр (int)
        Возвращает:
        q1 (ndarray): Хорошие ключевые точки соответствия положения на i-1-м изображении
        q2 (ndarray): Хорошие ключевые точки соответствия положения на i изображении
        """
        # Нахождение ключевых точек и дескрипторов с помощью ORB
        kp1, des1 = self.orb.detectAndCompute(self.images[i - 1], None)
        kp2, des2 = self.orb.detectAndCompute(self.images[i], None)
        # Нахождение совпадений, где k число лучших совпадений для каждого дескриптора
        matches = self.flann.knnMatch(des1, des2, k=2)

        # Нахождение совпадений, которые не имеют большого расстояния
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        draw_params = dict(matchColor = -1, # обрисовкасоединяющих линий зеленым цветом
                 singlePointColor = None, #Цвет отдельных ключевых точек (кружков), что означает, что ключевые точки не имеют совпадений
                 matchesMask = None, # Маска, определяющая, какие совпадения будут нарисованы. Если маска пуста, все совпадения отображаются
                 flags = 2) #Флаги, устанавливающие функции рисования

        '''
        Рисует найденные совпадения ключевых точек из двух изображений.
        images[i], images[i-1] - первое и второе исходное изображение
        kp1, kp2 - ключевые точки из первого и второго исходного изображения
        good - список точек соответствия первого и воторого изображения
        outImg - вывод изображения
        '''

        img3 = cv2.drawMatches(self.images[i], kp1, self.images[i-1],kp2, good ,outImg = None,**draw_params)

        # Меняет размер drawMatches на размер одного изображения
        print(self.images[i].shape)
        height, width = self.images[i].shape
        img3 = cv2.resize(img3, (width, height))

        cv2.imshow("image", img3)
        cv2.waitKey(100)

        # Получение списока точек соответствия первого и воторого изображения
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2

    def get_pose(self, q1, q2):
        """
        Функция вычисляет матрицу преобразования
        Параметры:
        q1 (ndarray): Хорошие ключевые точки соответствия положения на i-1-м изображении
        q2 (ndarray): Хорошие ключевые точки соответствия положения на i изображении
        Возращает:
        transformation_matrix (ndarray) - Матрица преобразования
        """
        # Вычисляет существенную матрицу из  точек на двух изображениях.
        #threshold - максимальное расстояние от точки до эпиполярной линии в пикселях, за пределами которого точка считается выбросом
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)

        # Разложение существенной матрицы на ветор перемещения и матрицу поворота
        R, t = self.decomp_essential_mat(E, q1, q2)

        # Получаем матрицу преобразования
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

    def decomp_essential_mat(self, E, q1, q2):
        """
        Функция разложиения существенной матрицы
        Параметры:
        E (ndarray): Существенная матрица
        q1 (ndarray): Хорошие ключевые точки соответствия положения на i-1-м изображении
        q2 (ndarray): Хорошие ключевые точки соответствия положения на i изображении
        Возращает:
        right_pair (list): Содержит матрицу поворота и вектор перемещения
        """
        def sum_z_cal_relative_scale(R, t):
            # Получение матрицы преобразования
            T = self._form_transf(R, t)
            # Создание проекционной матрицы
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Триангуляция 3D-точек
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = np.matmul(T, hom_Q1)

            # Не гомогенизировать
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Поиск количества точек, имеющих положительную координату z в обеих камерах
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Формирование пары точек и вычисление относительного масштаба
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        '''
        Разложиение существенной матрицы
        R1, R2 - первая и вторая из возможных матриц вращения
        t - один из возможных вариантов перевода
        '''

        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t) #удаляется лишняя ось в массиве

        # Составление списока различных возможных пар
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Определение правильного решения
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Отбор пары, в которой больше всего точек с положительной координатой z
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]

def haversine(dists,angles):
    """Функция для нахождения координаты второй точки по углу и расстоянию от первой
    Параметры:
        dists: массив значений(метрах)отрображающих расстояние между в ряд идущими точками 
        angles: массив значений(градусах) в ряд идущих азимутов
    Возращает:
        Px,Py: GPS координаты в 2D, по оси Ох и Оу 
    """
    Px=[0] #координаты по оси X
    Py=[0] #координаты по оси Y
    for i in range(len(dists)):
        Px.append(Px[i]+dists[i]*math.cos(angles[i]-angles[0]))
        Py.append(Py[i]+dists[i]*math.sin(angles[i]-angles[0]))
    return Px,Py

def main():
    data_dir = '../data/city'
    vo = VisualOdometry(data_dir)

    orb_x = []#массив позиций по x
    orb_y = []#массив позиций по y
    for i, gt_pose in enumerate(vo.gt_poses): #передается массив с позицией, i - индекс
        if i == 0:
            cur_pose = gt_pose
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            transf = np.nan_to_num(transf, neginf=0,posinf=0)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf)) #вычисляем обратную матрицу transf и находим произведение с cur_pose
        orb_x.append(cur_pose[0, 3])
        orb_y.append(cur_pose[2, 3])
 
    #gps_x,gps_y = haversine(vo.dists,vo.angles)

    #Выводим графики GPS и ORB
    plt.figure(1)
    #plt.plot(np.array(gps_y),np.array(gps_x), label="GPS", color='g')
    plt.legend()
    plt.figure(2)
    plt.plot(orb_x,orb_y, label="ORB", color='r')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
