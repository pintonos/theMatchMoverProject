import cv2
import numpy as np

'''
Functions for stereo pose estimation
'''


def get_P(R, t, K):
    Rt = np.c_[R, t]
    return np.dot(K, Rt)


def get_front_of_camera(svd, K, points_3d):
    R1 = svd[0]
    R2 = svd[1]
    t = svd[2]

    P1 = get_P(R1, t, K)
    P2 = get_P(R2, t, K)

    p1_pos_checks = 0
    p2_pos_checks = 0
    for point_3d in points_3d:
        point_3d_hom = np.append(point_3d, np.asarray([1])).reshape(4, 1)

        w1 = np.matmul(P1, point_3d_hom)[2]
        M1 = np.delete(P1, np.s_[-1], axis=1)

        w2 = np.matmul(P2, point_3d_hom)[2]
        M2 = np.delete(P2, np.s_[-1], axis=1)

        if w1 * np.linalg.det(M1) > 0:
            p1_pos_checks += 1

        if w2 * np.linalg.det(M2) > 0:
            p2_pos_checks += 1

    if p1_pos_checks >= p2_pos_checks:
        R = R1
    else:
        R = R2

    sanity_check_R(R)
    return R, t


def get_F(pts1, pts2):
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 4, 0.999)
    F = F/np.linalg.norm(F)
    return F, mask


def get_E_from_F(pts1, pts2, K):
    # Get fundamental Matrix
    F, mask = get_F(pts1, pts2)

    # Compute E from F
    E = np.dot(np.dot(np.transpose(K), F), K)
    return E, mask


def filter_pts(pts1, pts2, mask):
    pts1_in = [pt for (pt, m) in zip(pts1, mask) if m[0] != 0]
    pts2_in = [pt for (pt, m) in zip(pts2, mask) if m[0] != 0]
    pts1 = np.reshape(pts1_in, (1, len(pts1_in), 2))
    pts2 = np.reshape(pts2_in, (1, len(pts2_in), 2))
    return pts1, pts2


def get_R_and_t(pts1, pts2, K, compute_with_f=False, own_cheirality_check=False):
    """
    get R and t from essential matrix E

    reference: https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
    """

    if compute_with_f:  # compute essential matrix via fundamental matrix
        E, mask = get_E_from_F(pts1, pts2, K)
    else:  # find essential matrix directly
        E, mask = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, prob=0.999, threshold=0.1, cameraMatrix=K)

    # sanity check for E
    if not np.isclose(np.linalg.det(E), 0.0, atol=1.e-3):
        raise Exception('det(E) != 0, instead it is:', np.linalg.det(E))

    # filter outliers
    if not own_cheirality_check:
        pts1, pts2 = filter_pts(pts1, pts2, mask)

        # refine mapping
        pts1, pts2 = cv2.correctMatches(E, pts1, pts2)

        # Recover relative camera rotation and translation from E and the corresponding points
        points, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

        sanity_check_R(R)
        return R, t

    svd = cv2.decomposeEssentialMat(E)
    return [svd[0], svd[1]], svd[2]


def triangulate_points(R1, t1, R2, t2, ref_pts1, ref_pts2, dist, K):
    """
    Get triangulated points of two given cameras and corresponding reference points. Reference points will be
    undistorted with K and dist.

    reference: https://stackoverflow.com/questions/16295551/how-to-correctly-use-cvtriangulatepoints/16299909
    """
    svd = None
    if type(R2) is list:
        svd = R2 + [t2]
        R2, t2 = svd[0], svd[2]

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

    if svd is not None:
        R2, t2 = get_front_of_camera(svd, K, world_coords)
        world_coords = triangulate_points(R1, t1, R2, t2, ref_pts1, ref_pts2, dist, K)

    return world_coords


def sanity_check_R(R):
    if not np.isclose(np.linalg.det(R), 1.0, atol=1.e-5):
        raise Exception('det(R) != 1, instead it is:', np.linalg.det(R))