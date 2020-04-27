from functions import *
from util import *
import cv2
import pandas as pd

# Check intermediate values
if K is None or dist is None:
    raise Exception('Camera matrix or distortion coefficient not found')

# get point correspondences
pts1 = pd.read_csv('../' + REF_POINTS_0, sep=',', header=None, dtype=float).values
pts2 = pd.read_csv('../' + REF_POINTS_100, sep=',', header=None, dtype=float).values

# get R and t
R1, t1 = INIT_ORIENTATION, INIT_POSITION
R2, t2 = get_R_and_t(pts1, pts2, K)

# get world points of axis
world_coords_axis, _ = get_3d_world_points(R1, t1, R2, t2, pts1, pts2, dist, K)

# project points to image 1
r_vec1, _ = cv2.Rodrigues(R1)
img_points1_2d, _ = cv2.projectPoints(world_coords_axis, r_vec1, t1, K, dist)

# project points to image 2
r_vec2, _ = cv2.Rodrigues(R2)
img_points2_2d, _ = cv2.projectPoints(world_coords_axis, r_vec2, t2, K, dist)


# Read images
img_1 = cv2.imread('../' + DATA_PATH + 'img_0.jpg')
img_2 = cv2.imread('../' + DATA_PATH + 'img_100.jpg')

# draw reference points
draw_points(img_1, pts1.astype(int))
draw_points(img_2, pts2.astype(int))

# show images
plot_show_img(img_1, img_points1_2d, REF_POINTS_0, axis=True)
plot_show_img(img_2, img_points2_2d, REF_POINTS_100, axis=True)

cv2.waitKey(0)

