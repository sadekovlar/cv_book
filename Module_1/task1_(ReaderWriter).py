import cv2
import os


if __name__ == "__main__":
    # check the current working directory (CWD)
    # run examples from root (cv_book folder)
    cwd = os.getcwd()

    file_path = os.path.join('data', 'road.png')
    input_image = cv2.imread(file_path)
    if input_image is None:
        print('CWD:', cwd)
        print('Image is empty. Check a file path')
        raise SystemExit(-100)

    cv2.imshow('Hello World', input_image)
    # if this function is called without parameters, the program will wait
    # for any button to be pressed
    k = cv2.waitKey()

    if ord('s') == k:  # S stands for Save
        cv2.imwrite('output.jpg', input_image)
        cv2.imwrite('output.png', input_image)

    cv2.destroyAllWindows()

    print(input_image.shape)

    print('Height of Image:', int(input_image.shape[0]), 'pixels')
    print('Width of Image: ', int(input_image.shape[1]), 'pixels')
    print('Number of color channels (depth): ', int(input_image.shape[2]),
          'channels')
