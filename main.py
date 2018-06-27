# reads image / video and displays results
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from utils import *
from calibration import *

PATH = 'test2.jpg'
INPUT_SIZE = (960, 540)

# read image
img = cv2.imread(PATH)

# convert to input size
img = cv2.resize(img, INPUT_SIZE)

# convert BGR to HLS space
img_hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

# combine both masks using bitwise OR
yellow_mask, white_mask = get_masks(img_hsl)
combined_mask = cv2.bitwise_or(yellow_mask, white_mask)

# apply mask to image using bitwise AND
img_masked = cv2.bitwise_and(img, img, mask=combined_mask)
show(img_masked)

# grayscale
img_gray = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)

# and blur for edge detection
kernel_size = 5
img_gray_blur = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)

# canny edge detection
low_thresh, high_thresh = 50, 150
img_canny = cv2.Canny(img_gray_blur, low_thresh, high_thresh)

# get only ROI by masking
img_roi = get_ROI(img_canny)

# hough to recognise straight lines
hough_lines = cv2.HoughLinesP(img_roi,
                                rho=1,
                                theta=(np.pi / 180.0),
                                threshold=15,
                                minLineLength=20,
                                maxLineGap=10)

left_lane_markings, right_lane_markings = diff_left_right(img, hough_lines)

result = trace_both_markings(img, left_lane_markings, right_lane_markings)

show(result)