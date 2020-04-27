from functions import *
from util import *
import cv2


# get world points of axis
axis = get_3d_axis()

# change sign of vector 0 to 3
vec_0_3 = axis[3] - axis[0]
vec_0_3 = - vec_0_3
axis[3] = vec_0_3 + axis[0]

# construct vector 0 to 4
vec_0_4 = (axis[3] - axis[0]) + (axis[2] - axis[0])
p4 = vec_0_4 + axis[0]

# construct vector 0 to 5
vec_0_5 = (axis[3] - axis[0]) + (axis[1] - axis[0])
p5 = vec_0_5 + axis[0]

# construct vector 0 to 6
vec_0_6 = (axis[1] - axis[0]) + (axis[2] - axis[0])
p6 = vec_0_6 + axis[0]

# construct vector 0 to 7
vec_2_7 = (p4 - axis[2]) + (p6 - axis[2])
p7 = vec_2_7 + axis[2]

# add new points to axis to get cube
cube = np.vstack((axis, p4))
cube = np.vstack((cube, p5))
cube = np.vstack((cube, p6))
cube = np.vstack((cube, p7))

# project points to image 1
r_vec1, _ = cv2.Rodrigues(INIT_ORIENTATION)
img_points1_2d, _ = cv2.projectPoints(cube, r_vec1, INIT_POSITION, K, dist)

# Read images
img_1 = cv2.imread(DATA_PATH + 'img_0.jpg')

# draw additional points of cube
drawpoints = []
for img in img_points1_2d:
    drawpoints.append(img[0])
draw_points(img_1, drawpoints)

cv2.imshow('cube', img_1)

cv2.waitKey(0)

