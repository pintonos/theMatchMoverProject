from util import *
import pandas as pd
from functions import *

""" Functions to draw points and shapes into an image
"""


def draw_cube(img, pts):
    # bottom
    cv2.line(img, (pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]), (255, 0, 0), 2)
    cv2.line(img, (pts[0][0], pts[0][1]), (pts[2][0], pts[2][1]), (255, 0, 0), 2)
    cv2.line(img, (pts[1][0], pts[1][1]), (pts[6][0], pts[6][1]), (255, 0, 0), 2)
    cv2.line(img, (pts[2][0], pts[2][1]), (pts[6][0], pts[6][1]), (255, 0, 0), 2)
    # top
    cv2.line(img, (pts[3][0], pts[3][1]), (pts[4][0], pts[4][1]), (255, 0, 0), 2)
    cv2.line(img, (pts[3][0], pts[3][1]), (pts[5][0], pts[5][1]), (255, 0, 0), 2)
    cv2.line(img, (pts[5][0], pts[5][1]), (pts[7][0], pts[7][1]), (255, 0, 0), 2)
    cv2.line(img, (pts[4][0], pts[4][1]), (pts[7][0], pts[7][1]), (255, 0, 0), 2)
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


def get_P(R, t, K):
    Rt = np.c_[R, t]
    return np.dot(K, Rt)


def get_3d_axis(R2, t2):

    pts1 = pd.read_csv(REF_POINTS_0, sep=',', header=None, dtype=float).values
    pts2 = pd.read_csv(REF_POINTS_18, sep=',', header=None, dtype=float).values

    P1 = get_P(INIT_ORIENTATION, INIT_POSITION, K)
    P2 = get_P(R2, t2, K)

    pts1 = pts1[:4]
    pts2 = pts2[:4]

    object_points = []
    for p1, p2 in list(zip(pts1, pts2)):
        ret = cv2.triangulatePoints(P1, P2, np.array([p1[0], p1[1]]), np.array([p2[0], p2[1]]))
        object_points.append(ret)
    object_points = cv2.convertPointsFromHomogeneous(np.array(object_points))

    return object_points
