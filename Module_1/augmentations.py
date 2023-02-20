import cv2
import numpy as np
from typing import List


def augmentation(paths: List[str], output_directory: str) -> None:
    """
    Функция делает аугметации указанных картинок (повороты во всех осях, блюр, отзеркаливание,

    :param paths: List of paths on src pictures
    :param output_directory: directory for new pictures
    :return: None
    """
    for path in paths:
        file_name = path.split("/")[-1]
        img = cv2.imread(path)
        cv2.imwrite(output_directory + '/' + file_name, img)

        # Добавление картинок с разными  цветами
        for i in range(3):
            for coef in range(1, 12, 3):
                img_first_dim_change = img.copy()
                img_first_dim_change[:, :, 0] = coef * img_first_dim_change[:, :, 0]
                for coef_2 in range(1, 12, 3):
                    img_second_dim_change = img_first_dim_change.copy()
                    img_second_dim_change[:, :, (i + 1) % 3] *= coef_2 * img_second_dim_change[:, :, (i + 1) % 3]
                    cv2.imwrite(output_directory + '/' + file_name + str(coef) + str(coef_2) + str(i) + '.jpg',
                                img_second_dim_change)

        # Повысил резкость
        kernel = np.array([[-1, -1, -1], [-1, 15, -1], [-1, -1, -1]])
        new_img = cv2.filter2D(img, -1, kernel)
        cv2.imwrite(output_directory + '/sharpen_' + file_name, new_img)


        # Повороты и отзеркаливания
        new_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(output_directory + '/rotated_cotnerclockwise_' + file_name, new_img)

        new_img = cv2.rotate(img, cv2.ROTATE_180)
        cv2.imwrite(output_directory + '/rotated_180_' + file_name, new_img)

        new_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(output_directory + '/rotated_clockwise_' + file_name, new_img)

        cv2.imwrite(output_directory + '/mirrored_' + file_name, np.flip(img, 1))

        # Блюр
        new_img = cv2.GaussianBlur(img, (21, 21), 0)
        cv2.imwrite(output_directory + '/blured' + file_name, new_img)


if __name__ == '__main__':
    images = ['./image.jpg']
    output = './output'

    augmentation(images, output)
