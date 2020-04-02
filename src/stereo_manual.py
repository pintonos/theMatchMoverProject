import cv2
import numpy as np
from matplotlib import pyplot as plt
from Constants import *
import pandas as pd


def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    center = tuple(imgpts[0].ravel())
    x = tuple(imgpts[1].ravel())
    y = tuple(imgpts[2].ravel())
    z = tuple(imgpts[3].ravel())
    xy = tuple(imgpts[4].ravel())
    xz = tuple(imgpts[5].ravel())
    yz = tuple(imgpts[6].ravel())
    xyz = tuple(imgpts[7].ravel())

    # draw middle in red
    cv2.line(img, x, xy, (0, 0, 255), 3)

    # draw top floor in blue
    cv2.line(img, y, xy, (0, 255, 0), 3)
    cv2.line(img, xy, xyz, (0, 255, 0), 3)
    cv2.line(img, xyz, yz, (0, 255, 0), 3)
    cv2.line(img, yz, y, (0, 255, 0), 3)

    # draw ground floor in green
    cv2.line(img, center, x, (255, 0, 0), 3)
    cv2.line(img, x, xz, (255, 0, 0), 3)
    cv2.line(img, xz, z, (255, 0, 0), 3)
    cv2.line(img, z, center, (255, 0, 0), 3)

    # draw middle in red
    cv2.line(img, center, y, (0, 0, 255), 3)
    cv2.line(img, xz, xyz, (0, 0, 255), 3)
    cv2.line(img, z, yz, (0, 0, 255), 3)


def draw_points(img, pts):
    for pt in pts:
        cv2.circle(img, (pt[0],pt[1]), 3, (0, 0, 255), -1)


def invert(R, t):
    backRotation = np.c_[R, t]
    backRotation = np.append(backRotation, [[0, 0, 0, 1]], axis=0)
    # https://answers.opencv.org/question/31421/opencv-3-essentialmatrix-and-recoverpose/
    backRotation = np.linalg.inv(backRotation)

    R_inv = backRotation[np.ix_([0,1,2],[0,1,2])]
    t_inv = backRotation[:,3][:-1].reshape(3,1)
    return R_inv, t_inv



# read manuel points
pts1 = pd.read_csv('../tmp/manual_pt1.csv', sep=',').values
pts2 = pd.read_csv('../tmp/manual_pt2.csv', sep=',').values

# Load previously saved data
K, dist = np.load(MAT_CAMERA), np.load(MAT_DIST_COEFF)

obj = np.float32([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]])

# project world coordinates to frame 1

t_vec_init = np.float32(np.asarray([1,1,-10])).reshape(3,1)
r_vec_init, _ = cv2.Rodrigues(np.identity(3))

imgpts1, _ = cv2.projectPoints(obj, r_vec_init, t_vec_init, K, dist)

# map to second image

# https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
# Normalize for Essential Matrix calaculation
pts_l_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1).astype(dtype=np.float32), cameraMatrix=K, distCoeffs=dist)
pts_r_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1).astype(dtype=np.float32), cameraMatrix=K, distCoeffs=dist)

E, _ = cv2.findEssentialMat(pts_l_norm, pts_r_norm, method=cv2.RANSAC, prob=0.999, threshold=0.1, cameraMatrix=K) # TODO test different settings

# recover relative camera rotation and translation from essential matrix and the corresponding points
points, R, t, _ = cv2.recoverPose(E, pts_l_norm, pts_r_norm)
print(points)
# IMPORTANT: R and t may need to be inverted:
# see https://answers.opencv.org/question/31421/opencv-3-essentialmatrix-and-recoverpose/
# and https://stackoverflow.com/questions/54486906/correctly-interpreting-the-pose-rotation-and-translation-after-recoverpose-f

# invert R and t
#R, t = invert(R, t)
r_vec, _ = cv2.Rodrigues(R, dst=dist)
t = np.add(t, t_vec_init)

# project world coordinates to frame 2
imgpts2, _ = cv2.projectPoints(obj, r_vec, t, K, dist)


# show images
img1 = cv2.imread('../data/img1.jpg')
img1 = cv2.resize(img1, (960, 540))
draw_points(img1, pts1)
draw(img1, imgpts1)
cv2.imshow('img1', img1)

img2 = cv2.imread('../data/img2.jpg')
img2 = cv2.resize(img2, (960, 540))
draw_points(img2, pts2)
draw(img2, imgpts2)
cv2.imshow('img2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()