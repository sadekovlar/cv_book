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

    if input_image.shape[2] > 1:
        blue, green, red = cv2.split(input_image)
        cv2.imshow('Blue',  blue)
        cv2.imshow('Green', green)
        cv2.imshow('Red',   red)

        k = cv2.waitKey()
        cv2.destroyAllWindows()

        if ord('s') == k:  # S stands for Save
            cv2.imwrite('blue.png',  blue)
            cv2.imwrite('green.png', green)
            cv2.imwrite('red.png', red)

        # you should pass list of images
        merged = cv2.merge([blue, green, red])
        cv2.imshow('Input',  input_image)
        cv2.imshow('Merged', merged)

        k = cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        print("The input has only one channel")
