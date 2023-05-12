import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def _load_images(filepath):
        """
        Функция создает массив изображений из видеозаписи
        Параметры:
        filepath (str): путь к каталогу с видео
        Возвращает:
        images (list): изображения
        """
        images = list()
        cap = cv2.VideoCapture(filepath)
        i=0 #переменная отслеживает каждый третий кадр
        while cap.isOpened():
            succeed, frame = cap.read()
            if succeed:
                if i == 5: #в массив изображений добавляется лишь каждый пятый кадр
                    images.append(frame)
                    i=0
                else: i+=1
                images.append(frame)
            else:
                cap.release()       
        return np.array(images)

def calibrate_left(calib_df, results_dict):

	results_dict["left_rms"], results_dict["left_camera_matrix"], results_dict["left_dist"], left_rotations, left_translations = cv2.calibrateCamera(
		objectPoints = calib_df["object_points"].to_numpy(), 
		imagePoints = calib_df["left_corners"].to_numpy(), 
		imageSize = results_dict["DIM"],
		cameraMatrix = np.zeros((3, 3)),
		distCoeffs = np.zeros((4, 1)),
		rvecs = calib_df["left_rotations"].to_numpy(),
		tvecs = calib_df["left_translations"].to_numpy(),
		flags = results_dict["left_right_flags"]
		)  
	# Обновляем результатами калибровки
	calib_df["left_rotations"].loc[list(calib_df.index)] = left_rotations
	calib_df["left_translations"].loc[list(calib_df.index)] = left_translations

	print("\nLeft Camera RMS: {}".format(results_dict["left_rms"]))
	print("\nLeft Camera Matrix: \n{}".format(results_dict["left_camera_matrix"] ))
	print("\nLeft Distortion Coefficients: \n{}\n".format(results_dict["left_dist"]))

	# Ошибка повторного проецирования
	for image_id, row in calib_df.iterrows():
		left_reprojected_points, _ = cv2.projectPoints(
			objectPoints = row["object_points"], 
			rvec = row["left_rotations"], 
			tvec = row["left_translations"], 
			cameraMatrix = results_dict["left_camera_matrix"] , 
			distCoeffs = results_dict["left_dist"]
			)
		# Добавляем ошибки перепроецирования и точки в фрейм данных
		left_reprojected_points = left_reprojected_points.reshape(-1,2)
		calib_df["left_reprojection_errors"].loc[image_id] = row["left_corners"] - left_reprojected_points
		calib_df["left_reprojection_points"].loc[image_id] = left_reprojected_points
		# Находим ошибку, аналогичную среднеквадратичному значению для каждого изображения
		calib_df["left_error"].loc[image_id] = np.sqrt(np.sum(np.square(row["left_corners"] - left_reprojected_points)) / 49)
	return calib_df, results_dict
def calibrate_right(calib_df, results_dict):
	results_dict["right_rms"], results_dict["right_camera_matrix"], results_dict["right_dist"], right_rotations, right_translations = cv2.calibrateCamera(
		objectPoints = calib_df["object_points"].to_numpy(), 
		imagePoints = calib_df["right_corners"].to_numpy(), 
		imageSize = results_dict["DIM"],
		cameraMatrix = np.zeros((3, 3)),
		distCoeffs = np.zeros((4, 1)),
		rvecs = calib_df["right_rotations"].to_numpy(),
		tvecs = calib_df["right_translations"].to_numpy(),
		flags = results_dict["left_right_flags"]
		)

	# Обновляем результатами калибровки
	calib_df["right_rotations"].loc[list(calib_df.index)] = right_rotations
	calib_df["right_translations"].loc[list(calib_df.index)] = right_translations

	# Обновляем результатами калибровки
	print("\nRight Camera RMS: {}".format(results_dict["right_rms"]))
	print("\nRight Camera Matrix: \n{}".format(results_dict["right_camera_matrix"]))
	print("\nRight Distortion Coefficients: \n{}\n".format(results_dict["right_dist"]))

	# Ошибка повторного проецирования
	for image_id, row in calib_df.iterrows():
		right_reprojected_points, _ = cv2.projectPoints(
			objectPoints = row["object_points"], 
			rvec = row["right_rotations"], 
			tvec = row["right_translations"], 
			cameraMatrix = results_dict["right_camera_matrix"], 
			distCoeffs = results_dict["right_dist"]
			)
		# Добавляем ошибки перепроецирования и точки в фрейм данных
		right_reprojected_points = right_reprojected_points.reshape(-1,2)
		calib_df["right_reprojection_errors"].loc[image_id] = row["right_corners"] - right_reprojected_points
		calib_df["right_reprojection_points"].loc[image_id] = right_reprojected_points
		# Находим ошибку, аналогичную среднеквадратичному значению для каждого изображения
		calib_df["right_error"].loc[image_id] = np.sqrt(np.sum(np.square(row["right_corners"] - right_reprojected_points)) / 49)

	return calib_df, results_dict

def main(pathleft,pathright):
    # Путь к видеофайлам
    pathL = str(Path('../data/stereo/kem.001',pathleft))
    pathR = str(Path('../data/stereo/kem.001',pathright))

    Left_img = _load_images(pathL)
    Right_img = _load_images(pathR)

	# id для каждой пары изображений
    left_image_names = [i for i in range(len(Left_img))]
    right_image_names = [i for i in range(len(Right_img))]

    # Количество изображений
    num_images = len(Left_img)

	# Размер по H и W
    image_width = 1280 
    image_height = 720 

	# Размеры шахматной доски
    num_vertical_corners = 7 
    num_horizontal_corners = 7

    total_corners = num_vertical_corners * num_horizontal_corners
    
	# Трехмерные координаты точек шаблона в калибровке камеры
    '''Трехмерные углы используются для вычисления векторов поворота и перемещения углов
    в системах координат левой и правой камер относительно системы координат объектов
    (система координат шахматной доски).
    '''
    objp = np.zeros((num_vertical_corners * num_horizontal_corners, 1, 3), np.float32)
    objp[:,0, :2] = np.mgrid[0:num_vertical_corners, 0:num_horizontal_corners].T.reshape(-1, 2)
    objp = np.array([corner for [corner] in objp])


    calib_dict = {
	"image_id": [i for i in range(1, num_images + 1)],
	"left_image_name": left_image_names,
	"right_image_name": right_image_names,
	"found_chessboard": True, # True или False
	"left_corners": "", # 2d points шахматной доски в плоскости изображения
	"right_corners": "", # 2d points шахматной доски в плоскости изображения
	"object_points": [[objp] for i in range(1, num_images + 1)], # 3d point в реальном пространстве (для левого и правого изображений)
	"left_rotations": [np.zeros((1, 1, 3), dtype=np.float32) for i in range(1, num_images + 1)], 
	"right_rotations": [np.zeros((1, 1, 3), dtype=np.float32) for i in range(1, num_images + 1)], 
	"left_translations": [np.zeros((1, 1, 3), dtype=np.float32) for i in range(1, num_images + 1)], 
	"right_translations": [np.zeros((1, 1, 3), dtype=np.float32) for i in range(1, num_images + 1)], 
	"left_reprojection_errors": "", # [x, y] ошибка на угол в пикселях
	"right_reprojection_errors": "", # [x, y] ошибка на угол в пикселях
	"left_error": "", # Ааналогично rms, но для каждого изображения
	"right_error": "", # Аналогично rms, но для каждого изображения
	"left_reprojection_points": "", # [x, y] ошибка на угол в пикселях
	"right_reprojection_points": "" # [x, y] ошибка на угол в пикселях
    }

	# Создайте фрейм данных для результатов калибровки
    calib_df = pd.DataFrame(calib_dict)
    calib_df = calib_df.set_index("image_id")

	# Поиск углов шахматной доски
    '''Метод OpenCV findChessboardCorners SB используется вместо стандартного метода findChessboardCorners,
       поскольку он более точный. Любая пара изображений с неудачным определением углов будет автоматически удалена в конце поиска'''
    for image_id, row in calib_df.iterrows():
        print ('\nОбработка левого изображения: {}, правого изображения: {}'.format(row["left_image_name"], row["right_image_name"]))
        left_image = Left_img[row["left_image_name"]]
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_image = Right_img[row["right_image_name"]]
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        
		# Критерии уточнения обнаруженных углов
        checkerboard_flags = cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
        
        # Поиск
        left_retval, left_corners_array = cv2.findChessboardCornersSB(left_gray, (num_vertical_corners, num_horizontal_corners), checkerboard_flags)
        right_retval, right_corners_array = cv2.findChessboardCornersSB(right_gray, (num_vertical_corners, num_horizontal_corners), checkerboard_flags)
		
		# Если на одной из пар изображений не найдены углы
        if not (left_retval and right_retval):
            calib_df["found_chessboard"].loc[image_id] = False
            print("Не удалось найти углы!")
            continue

        left_corners_array = left_corners_array.reshape(-1,2)
        right_corners_array = right_corners_array.reshape(-1,2)

        calib_df["left_corners"].loc[image_id] = left_corners_array
        calib_df["right_corners"].loc[image_id] = right_corners_array
        calib_df["object_points"].loc[image_id] = objp

        # Удаляет все изображения, на которых не удалось определить углы
        calib_df.drop(calib_df[(calib_df["found_chessboard"] == False)].index, axis=0, inplace=True)

	# Создаем словарь для сохранения результатов калибровки
    results_dict = {
		"left_camera_matrix": "",
		"right_camera_matrix": "",
		"left_dist": "",
		"right_dist": "",
		"width": image_width,
		"height": image_height,
		"DIM": (image_width, image_height),
		"left_rms": "",
		"right_rms": "",	
		"left_map_x_undistort": "", 
		"left_map_y_undistort": "",
		"right_map_x_undistort": "", 
		"right_map_y_undistort": "",
		"left_right_flags": cv2.CALIB_ZERO_TANGENT_DIST,
		"left_right_criteria": (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
		# Стерео значения
		"stereo_rms": "",
		"R": "", 
		"T": "",
		"stereo_flags": cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_ZERO_TANGENT_DIST,
		"stereo_criteria": (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
		"left_RT": "", 
		"right_RT": "", 
		"left_P": "", 
		"right_P": "", 
		"Q": "",
		"left_map_x_rectify": "", 
		"left_map_y_rectify": "",
		"right_map_x_rectify": "", 
		"right_map_y_rectify": ""}
    
	# Баг на последней паре, удаляем её
    calib_df = calib_df.drop(calib_df.tail(1).index)  
    
	# Найдем левую внутреннюю матрицу и коэффициенты искажения
    calib_df, results_dict = calibrate_left(calib_df, results_dict)
    
	# Найдите правую внутреннюю матрицу и коэффициенты искажения
    calib_df, results_dict = calibrate_right(calib_df, results_dict)
    
	# Среднее значение левой и правой ошибок
    calib_df["mean_errors"] = calib_df.apply(lambda row: (row["left_error"] + row["right_error"]) / 2, axis=1)
    
	# Сохраняем изображения с максимальной среднеквадратичной производительностью (меньшее соотношение означает меньшее количество изображений)
    recalibrate = True      
    keep_best_ratio = 0.7 

    if recalibrate:
        # Находим и удаляем изображения с наихудшей производительностью
        mean_errors = calib_df["mean_errors"]
        mean_errors = mean_errors.sort_values()

        # Выбераем худшие изображения для удаления
        images_to_drop = list(mean_errors.iloc[int(len(mean_errors) * keep_best_ratio):].index)
        # Удаление некачественных изображений
        calib_df.drop(images_to_drop, axis=0, inplace=True)
        calib_df, results_dict = calibrate_left(calib_df, results_dict)
        calib_df, results_dict = calibrate_right(calib_df, results_dict)

    # Найдем вращение, R и перемещение T левой системы координат камеры относительно правой системы координат камеры.
    results_dict["stereo_rms"], _, _, _, _, results_dict["R"], results_dict["T"], E, F = cv2.stereoCalibrate(
		objectPoints = calib_df["object_points"].to_numpy(), 
		imagePoints1 = calib_df["left_corners"].to_numpy(), 
		imagePoints2 = calib_df["right_corners"].to_numpy(), 
		cameraMatrix1 = results_dict["left_camera_matrix"], 
		distCoeffs1 = results_dict["left_dist"], 
		cameraMatrix2 = results_dict["right_camera_matrix"], 
		distCoeffs2 = results_dict["right_dist"], 
		imageSize = results_dict["DIM"], 
		R = None, 
		T = None,
		flags = results_dict["stereo_flags"], 
		criteria = results_dict["stereo_criteria"]
		)
    
	# Левое и правое выпрямляющие преобразования и проекционные матрицы
    (results_dict["left_RT"], results_dict["right_RT"], results_dict["left_P"], results_dict["right_P"], results_dict["Q"], validPixROI1, validPixROI2) = cv2.stereoRectify(
		cameraMatrix1 = results_dict["left_camera_matrix"], 
		distCoeffs1 = results_dict["left_dist"], 
		cameraMatrix2 = results_dict["right_camera_matrix"], 
		distCoeffs2 = results_dict["right_dist"], 
		imageSize = results_dict["DIM"], 
		R = results_dict["R"], 
		T = results_dict["T"]
		)

	# Левую и правую карты, приведенные ниже, можно использовать для устранения искажений на изображениях. 
    results_dict["left_map_x_undistort"], results_dict["left_map_y_undistort"] = cv2.initUndistortRectifyMap(
	cameraMatrix = results_dict["left_camera_matrix"], 
	distCoeffs = results_dict["left_dist"], 
	R = None, 
	newCameraMatrix = None, 
	size = results_dict["DIM"], 
	m1type = cv2.CV_16SC2)

    results_dict["right_map_x_undistort"], results_dict["right_map_y_undistort"] = cv2.initUndistortRectifyMap(
        cameraMatrix = results_dict["right_camera_matrix"], 
        distCoeffs = results_dict["right_dist"], 
        R = None, 
        newCameraMatrix = None, 
        size = results_dict["DIM"], 
        m1type = cv2.CV_16SC2)
    
    results_dict["left_map_x_rectify"], results_dict["left_map_y_rectify"] = cv2.initUndistortRectifyMap(
	cameraMatrix = results_dict["left_camera_matrix"], 
	distCoeffs = results_dict["left_dist"], 
	R = results_dict["left_RT"], 
	newCameraMatrix = results_dict["left_P"], 
	size = results_dict["DIM"], 
	m1type = cv2.CV_16SC2 
	)

    results_dict["right_map_x_rectify"], results_dict["right_map_y_rectify"] = cv2.initUndistortRectifyMap(
        cameraMatrix = results_dict["right_camera_matrix"], 
        distCoeffs = results_dict["right_dist"], 
        R = results_dict["right_RT"], 
        newCameraMatrix = results_dict["right_P"], 
        size = results_dict["DIM"], 
        m1type = cv2.CV_16SC2 
    )

    index = 1

    resized_width = 1280
    resized_height = 480 
    # Загрузим изображение
    left_image = Left_img[0]
    right_image = Right_img[0]

	# Объединим левое и правое
    joined = np.concatenate([left_image, right_image], axis=1)
    
    joined_small = cv2.resize(joined, (resized_width, resized_height), interpolation = cv2.INTER_AREA)

    joined_small = cv2.cvtColor(joined_small, cv2.COLOR_BGR2RGB)
    cv2.imshow('Необработанные правое и левое изображения',joined_small)
    cv2.waitKey()

	# Применяем карты
    left_map_x_undistort = results_dict["left_map_x_undistort"]
    right_map_x_undistort = results_dict["right_map_x_undistort"]
    left_map_y_undistort = results_dict["left_map_y_undistort"]
    right_map_y_undistort = results_dict["right_map_y_undistort"]

    undistorted_left = cv2.remap(left_image, left_map_x_undistort, left_map_y_undistort, interpolation=cv2.INTER_LINEAR)
    undistorted_right = cv2.remap(right_image, right_map_x_undistort, right_map_y_undistort, interpolation=cv2.INTER_LINEAR)
    joined_undistort = np.concatenate([undistorted_left, undistorted_right], axis=1)
    joined_undistorted_small = cv2.resize(joined_undistort, (resized_width, resized_height), interpolation = cv2.INTER_AREA)

    joined_undistorted_small = cv2.cvtColor(joined_undistorted_small, cv2.COLOR_BGR2RGB)

    cv2.imshow('',joined_undistorted_small)
    cv2.waitKey()

    pass

if __name__ == '__main__':
    main(pathleft='kem.001.003.left.avi',pathright='kem.001.003.right.avi')
    pass