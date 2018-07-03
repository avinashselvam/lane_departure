import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# these are standard procedures to obtain distortion matrix of the camera
# and use them to undistort captured images

def calib_cam(nx, ny, basepath):

    # stores pts needed to find distrotion matrix
    objpoints = []
    imgpoints = []

    objp = np.zeros((nx*ny, 3))
    # replaces first 2 cols.
    # np.mgrid makes [2, nx, ny] ndarray (read documentation)
    # transpose to [ny, nx, 2] => [ny*nx, 2] 
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # go thorugh each img in calibration dir and find chessboard corners
    for path in os.listdir(basepath):
        img = cv2.imread(basepath + path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # if corners are found
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

    return objpoints, imgpoints

def get_undistorted (img):

    # call above function
    # change nx, ny as per your calibration images
    objpoints, imgpoints = calib_cam(9, 6, './calibration/')

    # use pts to find matrix and return undistorted image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _ , matrix, dist, _ , _ = cv2.calibrateCamera(objpoints, imgpoints, img_gray.shape[::-1], None, None)
    undistorted = cv2.undistort(img, matrix, dist, None, matrix)

    return undistorted

