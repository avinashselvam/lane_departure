"""
Used threshold values and code based on
https://github.com/kenshiro-o

A brief and good write-up on lane keeping by same guy
https://towardsdatascience.com/teaching-cars-to-see-advanced-lane-detection-using-computer-vision-87a01de0424f
"""

# reads image / video and displays results
import cv2
import matplotlib.pyplot as plt
import numpy as np

def show(img):
    plt.imshow(img[...,::-1])
    plt.show()

def compare(img, img_aug):
    plt.imshow(img[...,::-1]); plt.subplot(211)
    plt.imshow(img_aug[...,::-1]); plt.subplot(212)
    plt.show()

PATH = ''

# image

img = cv2.imread(PATH)
img_aug = predict_lane(img)

# video

cap = cv2.VideoCapture('vtest.avi')

while(cap.isOpened()):
    ret, frame = cap.read()

    img_aug = predict_lane(frame)

    compare(frame, img_aug)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

### The process

### remove image distortion

# Thresholding

# HLS thresholding first

# convert BGR to HLS space
img_hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

# get yellow and white mask
def get_yellow_mask(img_hsl, low_thresh = [15, 38, 115],
                        high_thresh = [35, 204, 255]): 
    yellow_mask = cv2.inRange(img_hsl, low_thresh, high_thresh)
    return yellow_mask

def get_white_mask(img_hsl, low_thresh = [0, 200, 0],
                        high_thresh = [180, 255, 255]): 
    white_mask = cv2.inRange(img_hsl, low_thresh, high_thresh)
    return white_mask

# combine both masks using bitwise OR
yellow_mask, white_mask = get_yellow_mask(img_hsl), get_white_mask(img_hsl)
combined_mask = cv2.bitwise_or(yellow_mask, white_mask)

# apply mask to image
img_masked = cv2.bitwise_and(img, img, mask=combined_mask)

# grayscale and blur for edge detection
img_gray = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)

kernel_size = 5
img_gray_blur = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)

# canny edge detection
low_thresh, high_thresh = 50, 150
img_canny = cv2.Canny(img_gray_blur, low_thresh, high_thresh)

def get_vertices_ROI(img):
    # depends on input resolution
    return []

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

    # apply mask on img and return
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

# Hough transform to recognise straight lines
img_hough = cv2.hough_transform(img_canny,
                                rho=1,
                                theta=(np.pi / 180.0),
                                threshold=15,
                                min_line_strength=20,
                                max_line_gap=10)
                            
# visualise hough lines
def draw_hough_lines(img, lines, color=[255, 0, 0], thickness=5):
    img_copy = img.copy()

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)
    
    return img_copy

# distinguishes between left and right markers on the lane
# uses slope information
def diff_left_right(img, lines):

    # get mid point along x axis
    img_width = img.shape[1]
    mid_x = img_width / 2.0

    # placeholders to contain left and right lane markings
    left_lane_markings = []
    right_lane_markings = []

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

# linear regression among all left pts and separately among all right pts







