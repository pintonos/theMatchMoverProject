import cv2
import numpy as np

""" Functions for stereo pose estimation
"""


def invert(R, t):
    back_rotation = np.c_[R, t]
    back_rotation = np.append(back_rotation, [[0, 0, 0, 1]], axis=0)

    # More explanation at https://answers.opencv.org/question/31421/opencv-3-essentialmatrix-and-recoverpose/
    back_rotation = np.linalg.inv(back_rotation)

    R_inv = back_rotation[np.ix_([0, 1, 2], [0, 1, 2])]
    t_inv = back_rotation[:, 3][:-1].reshape(3, 1)
    return R_inv, t_inv


def get_E_from_F(pts1, pts2, K):
    """ Currently unused
    """
    # Get fundamental Matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC)
    F = F/np.linalg.norm(F)

    # Compute E from F
    E = np.dot(np.dot(np.transpose(K), F), K)
    return E


def stereo_view_map(pts1, pts2, t_vec, K, dist, img_points_3d, obj_orientation):
    # More explanation at https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv

    # Normalize for Essential Matrix calculation
    # pts_l_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1).astype(dtype=np.float32), cameraMatrix=K, distCoeffs=dist)
    # pts_r_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1).astype(dtype=np.float32), cameraMatrix=K, distCoeffs=dist)

    E, _ = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, prob=0.999, threshold=0.1,
                                cameraMatrix=K)  # TODO test different settings

    # refine mapping
    pts1 = np.reshape(pts1, (1, len(pts1), 2))
    pts2 = np.reshape(pts2, (1, len(pts2), 2))
    pts1, pts2 = cv2.correctMatches(E, pts1, pts2)

    # Recover relative camera rotation and translation from E and the corresponding points
    points, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    # Project world coordinates to frame 2
    t = np.add(t, np.expand_dims(t_vec, axis=1))
    R = R @ obj_orientation
    r_vec, _ = cv2.Rodrigues(R, dst=dist)
    
    img_points_2d, _ = cv2.projectPoints(img_points_3d, r_vec, t, K, dist)
    return R, t.reshape(1, 3)[0], img_points_2d
