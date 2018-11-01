# lane_segmentation
segments out the lane the car is currently running on.

Steps involved (pretty naive) :

1. selected color spaces where white and yellow are prominent. Since lane markings of that color.
2. threshold and segment these white + yellow pixels.
3. Now combine this image to obtain a binary image.
4. Use a linear or quadratic polyfit on the white pixels to get a curve.
5. Define y_min and y_max to get boundaries.
