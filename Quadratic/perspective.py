import cv2
import numpy as np


def get_transform_coords(img):

    bottom_px = img.shape[0] - 1

    src_pts = np.array([[210,bottom_px], [595,450],
                        [690,450], [1110, bottom_px]], np.float32)
    dst_pts = np.array([[200, bottom_px], [200, 0],
                        [1000, 0], [1000, bottom_px]], np.float32)

    return src_pts, dst_pts


# def get_perspective_transform_matrices(src, dst):

#     M = cv2.getPerspectiveTransform(src, dst)
#     M_inv = cv2.getPerspectiveTransform(dst, src)

#     return M, M_inv

def get_warped(img):

    src, dst = get_transform_coords(img)

    M = cv2.getPerspectiveTransform(src, dst)
    img_size = img.shape[1], img.shape[0]
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped

def get_unwarped(img):

    src, dst = get_transform_coords(img)

    M = cv2.getPerspectiveTransform(dst, src)
    img_size = img.shape[1], img.shape[0]
    unwarped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return unwarped