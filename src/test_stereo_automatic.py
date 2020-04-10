from functions import *
from util import *

np.set_printoptions(suppress=True)

# Check intermediate values
if K is None or dist is None:
    raise Exception('Camera matrix or distortion coefficient not found')

# Points for a 3D cube
img_points_3d = get_3d_cube_points()

# Project world coordinates to first frame
r_vec_id, _ = cv2.Rodrigues(OBJECT_ORIENTATION)
t_vec = OBJECT_POSITION
proj_points_2d, _ = cv2.projectPoints(img_points_3d, r_vec_id, t_vec, K, dist)

# Read images
img_1 = cv2.imread('./' + DATA_PATH + 'img_1.jpg')
img_2 = cv2.imread('./' + DATA_PATH + 'img_2.jpg')
img_3 = cv2.imread('./' + DATA_PATH + 'img_3.jpg')

# Match points automatically
match_points_1, match_points_2 = get_points(img_1, img_2, filter=True, detector=Detector.FAST,
                                            matcher=Matcher.FLANN)
match_points_1_3, match_points_3 = get_points(img_1, img_3, filter=True, detector=Detector.FAST,
                                            matcher=Matcher.FLANN)

# Project points to 3d
R, t, proj_points_img_2 = stereo_view_map(match_points_1, match_points_2, t_vec, K, dist, img_points_3d,
                                          OBJECT_ORIENTATION)
print(t)
R, t, proj_points_img_3 = stereo_view_map(match_points_1_3, match_points_3, t, K, dist, img_points_3d, R)
print(t)

# Plot images
draw_points(img_1, match_points_1)
draw_points(img_2, match_points_2)
draw_points(img_3, match_points_3)

plot_show_img(img_1, proj_points_2d, 'img_1')
plot_show_img(img_2, proj_points_img_2, 'img_2')
plot_show_img(img_3, proj_points_img_3, 'img_3')

cv2.waitKey(0)
cv2.destroyAllWindows()
