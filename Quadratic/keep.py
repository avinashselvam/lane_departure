from calibration import get_undistorted
from sobel import get_combined_binary
from perspective import get_warped
import cv2
import matplotlib.pyplot as plt
import numpy as np

PATH = 'test2.jpg'
INPUT_SIZE = (1280, 720)

img = cv2.imread(PATH)
img = cv2.resize(img, INPUT_SIZE)

def show(img):
    plt.imshow(img, cmap='gray'); plt.show()

def binarise(img):
    ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    show(thresh)

# gets x coordinates where there a lot of whites
def half_histogram(img):
    height = img.shape[0]
    # compute sum of pixels columnwise
    histogram = np.sum(img[height//2:,:], axis=0)
    return histogram    

# show(img)

# undistorted = undistort_img(img)
masked = get_combined_binary(img)
warped = get_warped(masked)
# binarise(warped)

# show(masked)
show(warped)

hist = half_histogram(warped)

plt.plot(hist)
plt.show()

mid_X = 600

left, right = warped[:,:mid_X], warped[:,mid_X:]

show(left)

ys, xs = left.nonzero()

left_fit = np.polyfit(ys, xs, 2)

print(left_fit)

p = np.poly1d(left_fit)

plt.plot(range(200),p(range(200)))
plt.show()

plt.scatter(xs, ys); plt.show()

# # the lane polynomials
# ll, rl = self.compute_lane_lines(thres_img_psp)

# # the lane curvature
# # defined by the radius of the largest circle
# # to which lane lines are tangential  
# lcr, rcr, lco = self.compute_lane_curvature(ll, rl)

# drawn_lines = draw_lane_lines(thres_img_psp, ll, rl)        
# #plt.imshow(drawn_lines)

# drawn_lines_regions = draw_lane_lines_regions(thres_img_psp, ll, rl)
# #plt.imshow(drawn_lines_regions)

# drawn_lane_area = self.draw_lane_area(thres_img_psp, undist_img, ll, rl)        
# #plt.imshow(drawn_lane_area)

# drawn_hotspots = self.draw_lines_hotspots(thres_img_psp, ll, rl)

# combined_lane_img = self.combine_images(drawn_lane_area, drawn_lines, drawn_lines_regions, drawn_hotspots, undist_img_psp)
# final_img = self.draw_lane_curvature_text(combined_lane_img, lcr, rcr, lco)

# self.total_img_count += 1
# self.previous_left_lane_line = ll
# self.previous_right_lane_line = rl
        
# return final_img
