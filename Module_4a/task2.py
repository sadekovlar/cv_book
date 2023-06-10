import cv2
import numpy as np
 
def getMatrixHomography(pathImg1, pathImg2):
	# Загрузка изображений
	img1 = cv2.imread(pathImg1)
	img2 = cv2.imread(pathImg2)

	# Перевод в серое
	gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	# Обнаружение ключевых точек и их дескрипторов
	sift = cv2.SIFT_create()
	kp1, des1 = sift.detectAndCompute(gray1, None)
	kp2, des2 = sift.detectAndCompute(gray2, None)

	# Сопоставление точек
	matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
	matches = matcher.match(des1, des2)

	# Сортировка сопоставлений по расстоянию между дескрипторами
	matches = sorted(matches, key=lambda x: x.distance)

	# Определение характеристик RANSAC
	ransacReprojThreshold = 5.0
	maxIters = 2000
	confidence = 0.995

	# Выбор лучших сопоставлений при помощи метода RANSAC
	src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
	dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold)

	# Удаление выбросов
	matchesMask = mask.ravel().tolist()
	good = []
	for i, match in enumerate(matches):
		if matchesMask[i]:
			good.append(match)

	# Отображение сопоставлений
	img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	im_out = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))
	im_out_gray = cv2.cvtColor(im_out, cv2.COLOR_BGR2GRAY)

	im_out_gray = cv2.resize(im_out_gray, (img2.shape[1], img2.shape[0]))

	# Отображение результата
	cv2.imshow('Matches', img3)
	cv2.imshow('Perspective_img1', im_out)
	cv2.imshow('Source_img2', img2)

	# Функция оценки качества
	k = cv2.getGaussianKernel(3, -1)
	k = cv2.resize(k, (img2.shape[1], img2.shape[0]))
	k = k / k.max()

	abs_diff = cv2.absdiff(gray2, im_out_gray)
	gmae = np.sum(k*abs_diff)/(abs_diff.shape[0]*abs_diff.shape[1])
	quality = 1 - gmae/255
	print(f'Quality = {quality:.4f}')

	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return M