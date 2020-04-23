from functions import *
from util import *
import cv2
import pandas as pd

# Check intermediate values
if K is None or dist is None:
    raise Exception('Camera matrix or distortion coefficient not found')


def get_3d_world_points(R1, t1, R2, t2, ref_pts1, ref_pts2):
    # undistort ref points
    # second answer: https://stackoverflow.com/questions/16295551/how-to-correctly-use-cvtriangulatepoints/16299909
    pts_l_norm = cv2.undistortPoints(np.expand_dims(ref_pts1, axis=1).astype(dtype=np.float32), cameraMatrix=K,
                                     distCoeffs=dist)
    pts_r_norm = cv2.undistortPoints(np.expand_dims(ref_pts2, axis=1).astype(dtype=np.float32), cameraMatrix=K,
                                     distCoeffs=dist)

    # triangulate points to get real world coordinates
    P1 = np.c_[R1, t1]
    P2 = np.c_[R2, t2]
    world_coords = cv2.triangulatePoints(P1, P2, pts_l_norm, pts_r_norm)

    # sanity check
    x2 = P2 @ world_coords
    x2 = x2 / x2[2]
    x2 = x2[:2]
    # should be equal
    print(pts_r_norm[:4])
    print(x2.transpose()[:4])

    # from homogeneous to normal coordinates
    world_coords /= world_coords[3]
    world_coords = world_coords[:-1]

    world_coords = world_coords.transpose()
    axis = world_coords[0:4]  # first 4 points are axis

    return axis, world_coords


# get point correspondences
pts1 = pd.read_csv('../' + REF_POINTS_1, sep=',', header=None, dtype=float).values
pts2 = pd.read_csv('../' + REF_POINTS_50, sep=',', header=None, dtype=float).values
pts3 = pd.read_csv('../' + REF_POINTS_150, sep=',', header=None, dtype=float).values

# get R and t
R1, t1 = INIT_ORIENTATION, INIT_POSITION
R2, t2 = get_R_and_t(pts1, pts2, K)
R3, t3 = get_R_and_t(pts1, pts3, K)

# get world points of axis
world_coords_axis, _ = get_3d_world_points(R1, t1, R2, t2, pts1, pts2)

# project points to image 1
r_vec1, _ = cv2.Rodrigues(R1)
img_points1_2d, _ = cv2.projectPoints(world_coords_axis, r_vec1, t1, K, dist)

# project points to image 2
r_vec2, _ = cv2.Rodrigues(R2)
img_points2_2d, _ = cv2.projectPoints(world_coords_axis, r_vec2, t2, K, dist)

# project points to image 3
r_vec3, _ = cv2.Rodrigues(R3)
img_points3_2d, _ = cv2.projectPoints(world_coords_axis, r_vec3, t3, K, dist)

# Read images
img_1 = cv2.imread('../' + DATA_PATH + 'img_1.jpg')
img_2 = cv2.imread('../' + DATA_PATH + 'img_2.jpg')
img_3 = cv2.imread('../' + DATA_PATH + 'img_3.jpg')

# draw reference points
draw_points(img_1, pts1.astype(int))
draw_points(img_2, pts2.astype(int))
draw_points(img_3, pts3.astype(int))

# show images
plot_show_img(img_1, img_points1_2d, REF_POINTS_1, axis=True)
plot_show_img(img_2, img_points2_2d, REF_POINTS_50, axis=True)
plot_show_img(img_3, img_points3_2d, REF_POINTS_150, axis=True)
cv2.waitKey(0)
cv2.destroyAllWindows()
