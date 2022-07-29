import cv2

from detector_stable_key_points import *

SHOW_RES = True

if __name__ == "__main__":
    vid = cv2.VideoCapture(r'./data/tram/trm.169.007.avi')

    while (vid.isOpened()):
        ret, frame = vid.read()
        scale, depth = 1.4, 3
        if ret == True:
            pyrs = get_down_pyramids(frame, scale, depth)
            key_points = get_key_points(pyrs)
            imgs_shape = [pyr.shape for pyr in pyrs]
            stable_key_points = find_stable_points_on_pyr_scales(imgs_shape, key_points, scale)
            upscaled_points, mults = upscale_points(key_points, scale)
            res_imgs = draw_points(frame, upscaled_points, stable_key_points, mults)

            if SHOW_RES:
                for id, img in enumerate(res_imgs):
                    cv2.imshow(f'KeyPoints-{id}', img)
                    cv2.imwrite(f'KeyPoints-{id}.png', img)
                cv2.waitKey(0)
