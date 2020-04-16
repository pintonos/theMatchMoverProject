from functions import *
from util import *

# Check intermediate values
if K is None or dist is None:
    raise Exception('Camera matrix or distortion coefficient not found')

# Points for a 3D cube
img_points_3d = get_3d_axis()

# Read images
img_1 = cv2.imread('./' + DATA_PATH + 'img_1.jpg')
img_2 = cv2.imread('./' + DATA_PATH + 'img_2.jpg')

# Match points automatically
match_points_1, match_points_2 = get_points(img_1, img_2)

# project points to image 1
r_vec, _ = cv2.Rodrigues(INIT_ORIENTATION, dst=dist)
img_points1_2d, _ = cv2.projectPoints(img_points_3d, r_vec, INIT_POSITION, K, dist)

# project points to image 2
R, t = get_R_and_t(match_points_1, match_points_2, INIT_ORIENTATION, INIT_POSITION, K)
r_vec, _ = cv2.Rodrigues(R, dst=dist)
img_points2_2d, _ = cv2.projectPoints(img_points_3d, r_vec, t, K, dist)

# Plot images
#draw_points(img_1, match_points_1)
#draw_points(img_2, match_points_2)

plot_show_img(img_1, img_points1_2d, 'img_1', axis=True)
plot_show_img(img_2, img_points2_2d, 'img_2', axis=True)

cv2.waitKey(0)
cv2.destroyAllWindows()
