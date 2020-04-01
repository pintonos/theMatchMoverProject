import cv2
import numpy as np
from matplotlib import pyplot as plt
from Constants import *
import pandas as pd

np.set_printoptions(suppress=True)

# https://stackoverflow.com/questions/30207467/how-to-draw-3d-coordinate-axes-with-opencv-for-face-pose-estimation
def draw_axis(img, R, t, K):
    # unit is mm
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img


def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    cv2.line(img, tuple(corners[0]), tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    cv2.line(img, tuple(corners[1]), tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    cv2.line(img, tuple(corners[2]), tuple(imgpts[2].ravel()), (0, 0, 255), 5)


# define 3D shape
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
corners = np.intc([[858, 580], [858, 580], [858, 580]])

# read manuel points
pts1 = pd.read_csv('../tmp/manual_pt1.csv', sep=',').values
pts2 = pd.read_csv('../tmp/manual_pt2.csv', sep=',').values

# get Fundamental matrix
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
print("F:\n", F)

# Load previously saved data
K, dist = np.load(MAT_CAMERA), np.load(MAT_DIST_COEFF)

# https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
# Normalize for Essential Matrix calaculation
pts_l_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1).astype(dtype=np.float32), cameraMatrix=K, distCoeffs=dist)
pts_r_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1).astype(dtype=np.float32), cameraMatrix=K, distCoeffs=dist)

E, _ = cv2.findEssentialMat(pts_l_norm, pts_r_norm, method=cv2.RANSAC, prob=0.999, threshold=3.0) # TODO test different settings
print("E:\n", E)

# recover relative camera rotation and translation from essential matrix and the corresponding points
points, R, t, mask = cv2.recoverPose(E, pts1, pts2)

# IMPORTANT: R and t may need to be inverted:
# see https://answers.opencv.org/question/31421/opencv-3-essentialmatrix-and-recoverpose/
# and https://stackoverflow.com/questions/54486906/correctly-interpreting-the-pose-rotation-and-translation-after-recoverpose-f
#R = np.linalg.inv(R)
#t = np.flip(t)

print("R:\n", R)
print("t:\n", t)

img1 = cv2.imread('../data/img1.jpg')

# project points to 2D image
imgpts1, jac = cv2.projectPoints(axis, R, t, K, dist)

# draw lines in image
draw(img1, corners, imgpts1)
#draw_axis(img1, R, t, K)
cv2.imshow('img1', img1)
cv2.waitKey(0)


# points can be mapped from the first image to the second by the formula: p2 = R*p1 + t
# see opencv link above
imagepts2 = []
for imagept in imgpts1:
    imagept = np.append(imagept, 1)
    imgpt2 = np.add(np.matmul(R, imagept), t) # map from img1 to img2
    imagepts2.append(imgpt2)

img2 = cv2.imread('../data/img2.jpg')

# draw lines in image
draw(img2, corners, imgpts1)

cv2.imshow('img2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()