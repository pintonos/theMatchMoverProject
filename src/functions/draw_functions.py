from util import *

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


def get_3d_axis():
    """
    world frames of axis, triangulated between frame 0 and 100
    """
    return np.float32([[0.48004413, 0.21423042, 2.0474179], [0.50754553, 0.09758199, 2.1601412],
                       [0.8129302,  0.22362995, 1.9758732], [0.4802751,  0.24871479, 2.059794]])
