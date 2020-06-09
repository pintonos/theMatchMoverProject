import functools
import operator

from util import *
import pandas as pd
from functions import *

""" Functions to draw points and shapes into an image
"""


def draw_cube(img, pts):
    # bottom
    cv2.line(img, (pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]), (0, 255, 0), 2)
    cv2.line(img, (pts[0][0], pts[0][1]), (pts[2][0], pts[2][1]), (0, 255, 0), 2)
    cv2.line(img, (pts[1][0], pts[1][1]), (pts[6][0], pts[6][1]), (0, 255, 0), 2)
    cv2.line(img, (pts[2][0], pts[2][1]), (pts[6][0], pts[6][1]), (0, 255, 0), 2)
    # top
    cv2.line(img, (pts[3][0], pts[3][1]), (pts[4][0], pts[4][1]), (0, 0, 255), 2)
    cv2.line(img, (pts[3][0], pts[3][1]), (pts[5][0], pts[5][1]), (0, 0, 255), 2)
    cv2.line(img, (pts[5][0], pts[5][1]), (pts[7][0], pts[7][1]), (0, 0, 255), 2)
    cv2.line(img, (pts[4][0], pts[4][1]), (pts[7][0], pts[7][1]), (0, 0, 255), 2)
    # connections
    cv2.line(img, (pts[0][0], pts[0][1]), (pts[3][0], pts[3][1]), (255, 0, 0), 2)
    cv2.line(img, (pts[2][0], pts[2][1]), (pts[4][0], pts[4][1]), (255, 0, 0), 2)
    cv2.line(img, (pts[6][0], pts[6][1]), (pts[7][0], pts[7][1]), (255, 0, 0), 2)
    cv2.line(img, (pts[1][0], pts[1][1]), (pts[5][0], pts[5][1]), (255, 0, 0), 2)


def draw_axis(img, points):
    points = np.int32(points).reshape(-1, 2)
    cv2.line(img, tuple(points[0]), tuple(points[1]), (255, 0, 0), 3)
    cv2.line(img, tuple(points[0]), tuple(points[2]), (0, 255, 0), 3)
    cv2.line(img, tuple(points[0]), tuple(points[3]), (0, 0, 255), 3)


def draw_points(img, pts, i_init=0):
    for i, pt in enumerate(pts):
        cv2.circle(img, (pt[0], pt[1]), 3, (255, 0, 0), -1)
        cv2.putText(img, str(i + i_init), (pt[0], pt[1]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)


def draw_keyframe_inliers(start_idx, inliers_2d):
    for i, keyframe_id in enumerate(start_idx[1:]):
        img = get_frame(keyframe_id)
        draw_points(img, functools.reduce(operator.iconcat, inliers_2d[i].astype(int).tolist(), []))
        cv2.imshow('normal', cv2.resize(img, DEMO_RESIZE))
        cv2.waitKey(0)


def get_cube_points_from_axis_points(camera, axis):
    axis = np.squeeze(axis, axis=1)
    # vectors
    scale_factor_xy = 0.5
    vec_0_1 = axis[1] - axis[0]
    vec_0_2 = (axis[2] - axis[0])

    vec_0_3 = axis[3] - axis[0]
    axis[3] = vec_0_3 + axis[0]

    vec_0_4 = vec_0_3 + vec_0_2
    vec_0_5 = vec_0_3 + vec_0_1
    vec_0_6 = vec_0_1 + vec_0_2
    vec_2_7 = (vec_0_4 + axis[0] - axis[2]) + (vec_0_6 + axis[0] - axis[2])

    # points
    p4 = vec_0_4 + axis[0]
    p5 = vec_0_5 + axis[0]
    p6 = vec_0_6 + axis[0]
    p7 = vec_2_7 + axis[2]

    # add new points to axis to get cube
    cube = np.vstack((axis, p4))
    cube = np.vstack((cube, p5))
    cube = np.vstack((cube, p6))
    cube = np.vstack((cube, p7))

    # project points to image 1
    img_points_2d, _ = cv2.projectPoints(cube, camera.R_vec, camera.t, K, dist)

    # draw additional points of cube
    drawpoints = []
    for img in img_points_2d:
        drawpoints.append(list(np.int_(img[0])))

    return drawpoints


def get_P(R, t, K):
    Rt = np.c_[R, t]
    return np.dot(K, Rt)


def get_3d_axis(camera_start, start, camera_end, end):
    pts1 = pd.read_csv(REF_POINTS.format(frame=str(start)), sep=',', header=None, dtype=float).values
    pts2 = pd.read_csv(REF_POINTS.format(frame=str(end)), sep=',', header=None, dtype=float).values

    P1 = get_P(camera_start.R_mat, camera_start.t, K)
    P2 = get_P(camera_end.R_mat, camera_end.t, K)

    object_points = []
    for p1, p2 in list(zip(pts1, pts2)):
        ret = cv2.triangulatePoints(P1, P2, np.array([p1[0], p1[1]]), np.array([p2[0], p2[1]]))
        object_points.append(ret)
    object_points = cv2.convertPointsFromHomogeneous(np.array(object_points))

    return object_points
