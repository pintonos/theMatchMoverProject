from functions import *
from util import *
import cv2
import pandas as pd

np.set_printoptions(suppress=True)

# Check intermediate values
if K is None or dist is None:
    raise Exception('Camera matrix or distortion coefficient not found')

ref_points_1 = pd.read_csv('../' + REF_POINTS_1, sep=',').values
ref_points_2 = pd.read_csv('../' + REF_POINTS_2, sep=',').values

# Points for a 3D cube
img_points_3d = get_3d_axis()

# Project world coordinates to first frame
r_vec_id, _ = cv2.Rodrigues(OBJECT_ORIENTATION)
t_vec = OBJECT_POSITION
proj_points_2d, _ = cv2.projectPoints(img_points_3d, r_vec_id, t_vec, K, dist)

# Read images
img_1 = cv2.imread('../' + DATA_PATH + 'img_1.jpg')
img_2 = cv2.imread('../' + DATA_PATH + 'img_2.jpg')

# Match points automatically
match_points_1, match_points_2 = get_points(img_1, img_2, filter=True, detector=Detector.FAST,
                                            matcher=Matcher.FLANN)

# Project points to 3d
R, t, proj_points_img_2 = stereo_view_map(match_points_1, match_points_2, img_points_3d, t_vec, OBJECT_ORIENTATION, K,
                                          dist)

P1 = K @ (np.c_[OBJECT_ORIENTATION, t_vec])
P2 = K @ np.c_[R, t]

ref_points_1 = np.array([[1334.0, 1335.0, 1590.0, 1613.0, 1610.0, 1332.0],
                         [654.0, 561.0, 574.0, 669.0, 697.0, 680.0]])
ref_points_2 = np.array([[1187.0, 1243.0, 1518.0, 1484.0, 1482.0, 1188.0],
                         [628.0, 521.0, 563.0, 679.0, 709.0, 654.0]])

world_coords = cv2.triangulatePoints(P1[:3], P2[:3], ref_points_1, ref_points_2)
print(world_coords)
world_coords /= world_coords[3]
print(world_coords)

# Plot images
draw_points(img_1, match_points_1)
# draw_points(img_2, match_points_2)


plot_show_img(img_1, proj_points_2d, 'img_1', axis=True)
# plot_show_img(img_2, proj_points_img_2, 'img_2')


cv2.waitKey(0)
cv2.destroyAllWindows()
