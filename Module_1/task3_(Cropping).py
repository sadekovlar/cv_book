import cv2
import os


if __name__ == "__main__":
    # check the current working directory (CWD)
    # run examples from root (cv_book folder)
    cwd = os.getcwd()
    # Load our input image
    file_path = os.path.join('data', 'road.png')
    input_image = cv2.imread(file_path)
    if input_image is None:
        print('CWD:', cwd)
        print('Image is empty. Check a file path')
        raise SystemExit(-100)

    height, width = input_image.shape[:2]

    start_row = int(height*.25)
    start_col = int(width*.25)

    end_row = int(height*.80)
    end_col = int(width*.80)

    cropped = input_image[start_row:end_row, start_col:end_col]

    cv2.imshow("original_image", input_image)
    cv2.waitKey(0)
    cv2.imshow("cropped", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
