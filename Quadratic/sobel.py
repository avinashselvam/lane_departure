# Has 2 parts - color thresholding, gradient thresholding
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Color thresholding

# get mask that covers yellow and white region
def get_color_mask(img):

    # good results for white & yellow from HSL space
    img_hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
    low_thresh = np.array([15, 30, 115])
    high_thresh = np.array([35, 204, 255]) 
    yellow_mask = cv2.inRange(img_hsl, low_thresh, high_thresh)

    low_thresh = np.array([0, 200, 0])
    high_thresh = np.array([180, 255, 255])
    white_mask = cv2.inRange(img_hsl, low_thresh, high_thresh)

    # bitwise or to combine the masks of both color
    combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
    
    return combined_mask

# Gradient thresholding with sobel operator

# interested in L channel of LAB space
def get_L(img):

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_l = img_lab[:,:,0]

    return img_l

def abs_sobel(img_l, kernel_size=3, thresh=(0, 255)):
    
    sobel_x = cv2.Sobel(img_l, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(img_l, cv2.CV_64F, 0, 1, ksize=kernel_size)

    s = []
    
    for sobel in [sobel_x, sobel_y]:
        sobel_abs = np.absolute(sobel)
        sobel_scaled = np.uint8(255 * sobel / np.max(sobel_abs))
        
        gradient_mask = np.zeros_like(sobel_scaled)
        gradient_mask[(thresh[0] <= sobel_scaled) & (sobel_scaled <= thresh[1])] = 1

        s.append(gradient_mask)
    
    return s

def mag_sobel(img_l, kernel_size=3, thresh=(0, 255)):

    sx = cv2.Sobel(img_l, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sy = cv2.Sobel(img_l, cv2.CV_64F, 0, 1, ksize=kernel_size)
    
    sxy = np.sqrt(np.square(sx) + np.square(sy))
    scaled_sxy = np.uint8(255 * sxy / np.max(sxy))
    
    sxy_binary = np.zeros_like(scaled_sxy)
    sxy_binary[(scaled_sxy >= thresh[0]) & (scaled_sxy <= thresh[1])] = 1
    
    return sxy_binary

def dir_sobel(img_l, kernel_size=3, thresh=(0, np.pi/2)):
    
    sx_abs = np.absolute(cv2.Sobel(img_l, cv2.CV_64F, 1, 0, ksize=kernel_size))
    sy_abs = np.absolute(cv2.Sobel(img_l, cv2.CV_64F, 0, 1, ksize=kernel_size))
    
    dir_sxy = np.arctan2(sx_abs, sy_abs)

    sdir_binary = np.zeros_like(dir_sxy)
    sdir_binary[(dir_sxy >= thresh[0]) & (dir_sxy <= thresh[1])] = 1
    
    return sdir_binary

def get_all_sobels(img_l):

    sobx_best, soby_best = abs_sobel(img_l, kernel_size=15, thresh=(20, 120))

    sobxy_best = mag_sobel(img_l, kernel_size=15, thresh=(80, 200))

    sobdir_best = dir_sobel(img_l, kernel_size=3, thresh=(0, np.pi/2))

    return sobx_best, soby_best, sobxy_best, sobdir_best

def get_combined_sobels(img):

    img_l = get_L(img)
    
    sx, sy, sxy, sdir = get_all_sobels(img_l)

    combined = np.zeros_like(sdir)
    # Sobel X returned the best output so we keep all of its results. We perform a binary and on all the other sobels    
    combined[(sx == 1) | ((sy == 1) & (sxy == 1) & (sdir == 1))] = 1
    
    return combined

# applying

def get_combined_binary(img):

    combined_color_mask = get_color_mask(img)
    combined_sobels = get_combined_sobels(img)
    
    print(combined_color_mask.dtype)
    print(combined_sobels.dtype)

    # bitwise or to combine results of both color and gradient
    combined_binary = cv2.bitwise_or(combined_color_mask, combined_sobels.astype(np.uint8))

    return combined_binary

