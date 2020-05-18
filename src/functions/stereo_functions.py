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


def get_F(pts1, pts2):
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 4, 0.999)
    F = F/np.linalg.norm(F)
    return F


def get_E_from_F(pts1, pts2, K):
    # Get fundamental Matrix
    F = get_F(pts1, pts2)

    # Compute E from F
    E = np.dot(np.dot(np.transpose(K), F), K)
    return E


def get_R_and_t(pts1, pts2, K, compute_with_f=False):
    # More explanation at https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv

    E = None
    if compute_with_f:  # compute essential matrix via fundamental matrix
        E = get_E_from_F(pts1, pts2, K)
    else:  # find essential matrix directly
        E, _ = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, prob=0.999, threshold=0.1, cameraMatrix=K)

    # sanity check for E
    if not np.isclose(np.linalg.det(E), 0.0, atol=1.e-3):
        raise Exception('det(E) != 0, instead it is:', np.linalg.det(E))

    # refine mapping
    pts1 = np.reshape(pts1, (1, len(pts1), 2))
    pts2 = np.reshape(pts2, (1, len(pts2), 2))
    pts1, pts2 = cv2.correctMatches(E, pts1, pts2)

    # Recover relative camera rotation and translation from E and the corresponding points
    points, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    # sanity check for R
    if not np.isclose(np.linalg.det(R), 1.0, atol=1.e-5):
        raise Exception('det(R) != 1, instead it is:', np.linalg.det(R))

    return R, t


def triangulate_points(R1, t1, R2, t2, ref_pts1, ref_pts2, dist, K):
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

    # from homogeneous to normal coordinates
    world_coords /= world_coords[3]
    world_coords = world_coords[:-1]

    world_coords = world_coords.transpose()

    return world_coords
