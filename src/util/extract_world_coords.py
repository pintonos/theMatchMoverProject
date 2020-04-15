from functions import *
from util import *
import cv2
import pandas as pd

# Check intermediate values
if K is None or dist is None:
    raise Exception('Camera matrix or distortion coefficient not found')

# Read images
img_1 = cv2.imread('../' + DATA_PATH + 'img_1.jpg')
img_2 = cv2.imread('../' + DATA_PATH + 'img_2.jpg')

# get R and t
pts1, pts2 = get_points(img_1, img_2)
R, t = get_R_and_t(pts1, pts2, OBJECT_POSITION, OBJECT_ORIENTATION, K)

# point correspondences of book
ref_points_1 = np.array([[1334.0, 1335.0, 1590.0, 1613.0, 1610.0, 1332.0],
                         [654.0, 561.0, 574.0, 669.0, 697.0, 680.0]])
ref_points_2 = np.array([[1187.0, 1243.0, 1518.0, 1484.0, 1482.0, 1188.0],
                         [628.0, 521.0, 563.0, 679.0, 709.0, 654.0]])

# triangulate points to get real world coordinates
P1 = K @ (np.c_[OBJECT_ORIENTATION, OBJECT_POSITION])
P2 = K @ np.c_[R, t]
world_coords = cv2.triangulatePoints(P1[:3], P2[:3], ref_points_1, ref_points_2)

# from homogeneous to normal coordinates
world_coords /= world_coords[3]
world_coords = world_coords[:-1]
# delete columns not needed for drawing axis
world_coords = np.delete(world_coords, 2, 1)
world_coords = np.delete(world_coords, 3, 1)
print(world_coords)

# project points to image 1
r_vec, _ = cv2.Rodrigues(OBJECT_ORIENTATION, dst=dist)
img_points1_2d, _ = cv2.projectPoints(world_coords, r_vec, OBJECT_POSITION, K, dist)

# project points to image 2
r_vec, _ = cv2.Rodrigues(R, dst=dist)
img_points2_2d, _ = cv2.projectPoints(world_coords, r_vec, t, K, dist)

# draw reference points
ref_points_1_pts = pd.read_csv('../' + REF_POINTS_1, sep=',', header=None).values
ref_points_2_pts = pd.read_csv('../' + REF_POINTS_2, sep=',', header=None).values
draw_points(img_1, ref_points_1_pts)
draw_points(img_2, ref_points_2_pts)

# show images
plot_show_img(img_1, img_points1_2d, 'img_1', axis=True)
plot_show_img(img_2, img_points2_2d, 'img_2', axis=True)
cv2.waitKey(0)
cv2.destroyAllWindows()
