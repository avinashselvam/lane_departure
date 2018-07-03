import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from utils import *

cap = cv2.VideoCapture('test.flv')

INPUT_SIZE = (960, 540)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # convert to input size
    frame = frame[:900,100:-100]
    img = cv2.resize(frame, INPUT_SIZE)

    # convert BGR to HLS space
    img_hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # combine both masks using bitwise OR
    yellow_mask, white_mask = get_masks(img_hsl)
    combined_mask = cv2.bitwise_or(yellow_mask, white_mask)

    # apply mask to image using bitwise AND
    img_masked = cv2.bitwise_and(img, img, mask=combined_mask)

    # grayscale
    img_gray = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)

    # and blur for edge detection
    kernel_size = 5
    img_gray_blur = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)

    try:
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

        
        cv2.imshow('frame',result)

    except:
        pass

    # Display the resulting frame
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()