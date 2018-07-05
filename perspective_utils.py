import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from calibration_utils import calibrate_camera, undistort
from binarization_utils import binarize


def birdeye(img, verbose=False):
    """
    Apply perspective transform to input frame to get the bird's eye view.
    :param img: input color frame
    :param verbose: if True, show the transformation result
    :return: warped image, and both forward and backward transformation matrices
    """
    h, w = img.shape[:2]
    # src = np.float32([[w, h],
    #                   [0, h],
    #                   [0.4375 * w, h * 0.49],
    #                   [0.5625 * w, h * 0.49]])
    # dst = np.float32([[w, h],  # br
    #                   [0, h],  # bl
    #                   [0, 0],  # tl
    #                   [w, 0]])  # tr

    h, w = img.shape[:2]
    src = np.float32([[w, 480],
                      [0, 480],
                      [w * 3 / 8, h * 3 / 5],
                      [w * 5 / 8, h * 3 / 5]])
    dst = np.float32([[w * 0.6, h * 1.15],  # br
                      [0, h * 1.15],  # bl
                      [0, 0],  # tl
                      [w * 0.6, 0]])  # tr

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    if verbose:
        f, axarray = plt.subplots(1, 2)
        f.set_facecolor('white')
        axarray[0].set_title('Before perspective transform')
        axarray[0].imshow(img, cmap='gray')
        for point in src:
            axarray[0].plot(*point, '.')
        axarray[1].set_title('After perspective transform')
        axarray[1].imshow(warped, cmap='gray')
        for point in dst:
            axarray[1].plot(*point, '.')
        for axis in axarray:
            axis.set_axis_off()
        plt.show()

    return warped, M, Minv


if __name__ == '__main__':

    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')

    # show result on test images
    for test_img in glob.glob('test_images/*.jpg'):

        img = cv2.imread(test_img)

        img_undistorted = img

        img_binary = binarize(img_undistorted, verbose=False)

        img_birdeye, M, Minv = birdeye(cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB), verbose=True)


