from functions import *
from util import *
import pandas as pd
import cv2
import numpy as np

""" 
Functions to draw points and shapes into an image
"""


def draw_cube(img, pts):
    # bottom
    cv2.line(img, (pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]), (0, 255, 0), 2)
    cv2.line(img, (pts[0][0], pts[0][1]), (pts[2][0], pts[2][1]), (0, 255, 0), 2)
    cv2.line(img, (pts[1][0], pts[1][1]), (pts[6][0], pts[6][1]), (0, 255, 0), 2)
    cv2.line(img, (pts[2][0], pts[2][1]), (pts[6][0], pts[6][1]), (0, 255, 0), 2)
    # top
    cv2.line(img, (pts[3][0], pts[3][1]), (pts[4][0], pts[4][1]), (255, 0, 0), 2)
    cv2.line(img, (pts[3][0], pts[3][1]), (pts[5][0], pts[5][1]), (255, 0, 0), 2)
    cv2.line(img, (pts[5][0], pts[5][1]), (pts[7][0], pts[7][1]), (255, 0, 0), 2)
    cv2.line(img, (pts[4][0], pts[4][1]), (pts[7][0], pts[7][1]), (255, 0, 0), 2)
    # connections
    cv2.line(img, (pts[0][0], pts[0][1]), (pts[3][0], pts[3][1]), (0, 0, 255), 2)
    cv2.line(img, (pts[2][0], pts[2][1]), (pts[4][0], pts[4][1]), (0, 0, 255), 2)
    cv2.line(img, (pts[6][0], pts[6][1]), (pts[7][0], pts[7][1]), (0, 0, 255), 2)
    cv2.line(img, (pts[1][0], pts[1][1]), (pts[5][0], pts[5][1]), (0, 0, 255), 2)


def get_cube_points_from_axis_points(camera, axis_points):
    axis = np.squeeze(axis_points, axis=1).copy()

    vec_0_1 = axis[1] - axis[0]
    vec_0_2 = axis[2] - axis[0]
    vec_0_3 = axis[0] - axis[3]  # switch direction of vector

    axis[3] = vec_0_3 + axis[0]

    vec_0_4 = vec_0_3 + vec_0_2
    vec_0_5 = vec_0_3 + vec_0_1

    # points
    p4 = vec_0_4 + axis[0]
    p5 = vec_0_5 + axis[0]
    p6 = axis[4]

    vec_2_4 = p4 - axis[2]
    vec_2_6 = p6 - axis[2]
    p7 = vec_2_4 + vec_2_6 + axis[2]

    # add new points to axis to get cube
    cube = np.vstack((axis[:4], p4))
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


def get_3d_points_from_ref(camera_start, start, camera_end, end):
    pts1 = pd.read_csv(REF_POINTS.format(frame=str(start)), sep=',', header=None, dtype=float).values
    pts2 = pd.read_csv(REF_POINTS.format(frame=str(end)), sep=',', header=None, dtype=float).values

    pts_l_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1).astype(dtype=np.float32), cameraMatrix=K,
                                     distCoeffs=dist)
    pts_r_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1).astype(dtype=np.float32), cameraMatrix=K,
                                     distCoeffs=dist)

    # triangulate points to get real world coordinates
    P1 = np.c_[camera_start.R_mat, camera_start.t]
    P2 = np.c_[camera_end.R_mat, camera_end.t]
    world_coords = cv2.triangulatePoints(P1, P2, pts_l_norm, pts_r_norm)

    # from homogeneous to normal coordinates
    world_coords /= world_coords[3]
    world_coords = world_coords[:-1]

    world_coords = world_coords.transpose()

    return np.expand_dims(np.array(world_coords), axis=1)
