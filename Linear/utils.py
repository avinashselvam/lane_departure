"""
Used threshold values and code based on
https://github.com/kenshiro-o

A brief and good write-up on lane keeping by same guy
https://towardsdatascience.com/teaching-cars-to-see-advanced-lane-detection-using-computer-vision-87a01de0424f
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# display BGR images
def show(img):
    plt.imshow(img[...,::-1])
    plt.show()

# visualise lines
def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    img_copy = img.copy()

    for line in lines:
        x1, y1, x2, y2 = list(map(int,line))
        cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)
    
    return img_copy

# get mask that covers yellow and white region
def get_masks(img_hsl):
    
    low_thresh = np.array([15, 38, 115])
    high_thresh = np.array([35, 204, 255]) 
    yellow_mask = cv2.inRange(img_hsl, low_thresh, high_thresh)

    low_thresh = np.array([0, 200, 0])
    high_thresh = np.array([180, 255, 255])
    white_mask = cv2.inRange(img_hsl, low_thresh, high_thresh)
    
    return yellow_mask, white_mask

# depends on input resolution
# get coordinates of the trapeziodal Region of Interest
# this is by eye-balling I believe
def get_vertices_ROI(img):

    height, width = img.shape[0:2]

    vertices = None
    
    region_bottom_left = (130 , height - 1)
    region_top_left = (410, 330)
    region_top_right = (650, 350)
    region_bottom_right = (width - 30, width - 1)
    vertices = np.array([[region_bottom_left , region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)

    return vertices

# applies mask to obtain ROI from the original img
def get_ROI(img):

    # creating a blank mask of the same shape as img
    mask = np.zeros_like(img)

    # if img has > 1 depth, color is a tuple
    try:
        num_channels = img.shape[2]
        white = (255,)*num_channels
    # else color is a scalar
    except:
        white = 255
    
    # obtain 4 vertices of the trapezoid ROI
    vertices = get_vertices_ROI(img)

    # fill ROI polygon in mask with white i.e unmask ROI
    cv2.fillPoly(mask, vertices, white)

    # apply mask on img
    masked_img = cv2.bitwise_and(img, mask)
    
    return masked_img


# distinguishes between left and right markers on the lane
# uses slope information
def diff_left_right(img, lines):

    # get mid point along x axis
    img_width = img.shape[1]
    mid_x = img_width / 2.0

    # placeholders to contain left and right lane markings
    left_lane_markings = []
    right_lane_markings = []
    
    # check if each line is left or right and append accordingly
    for line in lines:
        for x1, y1, x2, y2 in line:
            
            dx, dy = x2 - x1, y2 - y1
            if dx == 0 or dy == 0:
                continue  # ignore 0 and 90 lines
            
            slope = dy/dx

            epsilon = 0.1 # arbitrary threshold
            if abs(slope) < epsilon:
                continue # ignore lines with extremly small slopes
            
            # marking should be entirely within left or right
            # if -ve slope left lane & +ve slope right lane
            # remember y axis downwards is +ve
            if slope < 0 and x1 < mid_x and x2 < mid_x:
                left_lane_markings.append(line)
            elif x1 >= mid_x and x2 >= mid_x:
                right_lane_markings.append(line) 

    return left_lane_markings, right_lane_markings


# given a set of points that make smaller line
# does linear regression to find line
# that best fits all the end points of small lines
def linreg(lines):
    xs, ys = [], []

    for line in lines:
        for x1, y1, x2, y2 in line:
            xs.append(x1)
            xs.append(x2)
            ys.append(y1)
            ys.append(y2)
    
    slope, intercept, _ , _ , _ = stats.linregress(xs, ys)

    return slope, intercept 

# given top_y we find the corresponding other coords with slope and intercept                        
def trace_lane_marking(img, lines, top_y):
    A, b = linreg(lines) # A is slope, b is intercept

    img_shape = img.shape
    bottom_y = img_shape[0] - 1 # zero indexing
    bottom_x = (bottom_y - b) // A # find bottom x intercept

    top_x = (top_y - b) // A # find top x intercept

    new_lines = [[bottom_x, bottom_y, top_x, top_y]]
    return draw_lines(img, new_lines)

# final output funtion that draws the predicted lane markings 
def trace_both_markings(img, left_lane_markings, right_lane_markings):
    vertices = get_vertices_ROI(img)
    _ , top_y = vertices[0][1]

    full_left_lane_img = trace_lane_marking(img, left_lane_markings, top_y)
    full_leftandright_lanes_img = trace_lane_marking(full_left_lane_img, right_lane_markings, top_y)

    img_with_lane_weight =  cv2.addWeighted(img, 0.5, full_leftandright_lanes_img, 0.5, 0.0)
    
    return img_with_lane_weight