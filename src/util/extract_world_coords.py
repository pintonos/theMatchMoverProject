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

    #print(P2)
    #print(world_coords.transpose()[0])
    #print(np.c_[np.linalg.inv(R2), -t2] @ world_coords.transpose()[0])

    # from homogeneous to normal coordinates TODO change to convertPointsFromHomogeneous
    world_coords /= world_coords[3]
    world_coords = world_coords[:-1]

    # delete columns not needed for drawing axis
    world_coords_axis = np.delete(world_coords, 2, 1)
    world_coords_axis = np.delete(world_coords_axis, 3, 1)
    world_coords_axis = world_coords_axis.transpose()
    world_coords = world_coords.transpose()

    return world_coords_axis, world_coords


# Read images
img_1 = cv2.imread('../' + DATA_PATH + 'img_1.jpg')
img_2 = cv2.imread('../' + DATA_PATH + 'img_2.jpg')
img_3 = cv2.imread('../' + DATA_PATH + 'img_3.jpg')

pts1 = pd.read_csv('../' + REF_POINTS_1, sep=',', header=None, dtype=float).values
pts2 = pd.read_csv('../' + REF_POINTS_50, sep=',', header=None, dtype=float).values
pts3 = pd.read_csv('../' + REF_POINTS_150, sep=',', header=None, dtype=float).values

# first 6 points are axis
ref_points_1 = pts1[:6]
ref_points_2 = pts2[:6]

# get R and t
R2, t2 = get_R_and_t(pts1, pts2, INIT_ORIENTATION, INIT_POSITION, K)
R3, t3 = get_R_and_t(pts1, pts3, INIT_ORIENTATION, INIT_POSITION, K)

world_coords_axis, _ = get_3d_world_points(INIT_ORIENTATION, INIT_POSITION, R2, t2, ref_points_1, ref_points_2)
print(world_coords_axis)

# project points to image 1
r_vec, _ = cv2.Rodrigues(INIT_ORIENTATION, dst=dist)
img_points1_2d, _ = cv2.projectPoints(world_coords_axis, r_vec, INIT_POSITION, K, dist)

# project points to image 2
r_vec2, _ = cv2.Rodrigues(R2, dst=dist)
img_points2_2d, _ = cv2.projectPoints(world_coords_axis, r_vec2, t2, K, dist)

# project points to image 3
r_vec3, _ = cv2.Rodrigues(R3, dst=dist)
img_points3_2d, _ = cv2.projectPoints(world_coords_axis, r_vec3, t3, K, dist)

# draw reference points
draw_points(img_1, pts1.astype(int))
draw_points(img_2, pts2.astype(int))
draw_points(img_3, pts3.astype(int))

# show images
plot_show_img(img_1, img_points1_2d, 'img_1', axis=True)
plot_show_img(img_2, img_points2_2d, 'img_2', axis=True)
plot_show_img(img_3, img_points3_2d, 'img_3', axis=True)
cv2.waitKey(0)
cv2.destroyAllWindows()
