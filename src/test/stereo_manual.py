import pandas as pd

from src.functions.draw_functions import *
from src.functions.stereo_functions import *
from src.util.config import *

""" Stereo image calibration with manual extracted points

Uses manual matched points to estimate the pose in a second frame.
"""

# Check intermediate values
if K is None or dist is None:
    raise Exception('Camera matrix or distortion coefficient not found')

# Read manual points
# Scale *2, points were matched in resized img
match_points_1 = pd.read_csv('../' + MANUAL_MATCH_POINTS_1, sep=',').values * 2
match_points_2 = pd.read_csv('../' + MANUAL_MATCH_POINTS_2, sep=',').values * 2
match_points_3 = pd.read_csv('../' + MANUAL_MATCH_POINTS_3, sep=',').values * 2

# Points for a 3D cube
img_points_3d = get_3d_cube_points()

# Project world coordinates to frame 1
r_vec_id, _ = cv2.Rodrigues(np.identity(3))
t_vec = np.float32(np.asarray([0, 0, SCALING_FACTOR]))
proj_points_img_1, _ = cv2.projectPoints(img_points_3d, r_vec_id, t_vec, K, dist)

# Map points to second and third image
proj_points_img_2 = stereo_view_map(match_points_1, match_points_2, t_vec, K, dist, img_points_3d)
proj_points_img_3 = stereo_view_map(match_points_1, match_points_3, t_vec, K, dist, img_points_3d)

# Read and plot images
img_1 = cv2.imread('../' + DATA_PATH + 'img_1.jpg')
img_2 = cv2.imread('../' + DATA_PATH + 'img_2.jpg')
img_3 = cv2.imread('../' + DATA_PATH + 'img_3.jpg')

draw_points(img_1, match_points_1)
draw_points(img_2, match_points_2)
draw_points(img_3, match_points_3)

plot_show_img(img_1, proj_points_img_1, 'img_1')
plot_show_img(img_2, proj_points_img_2, 'img_2')
plot_show_img(img_3, proj_points_img_3, 'img_3')

cv2.waitKey(0)
cv2.destroyAllWindows()