import cv2
import numpy as np
from matplotlib import pyplot as plt
from Constants import *
import pandas as pd


def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    center = (464, 302)
    cv2.line(img, center, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    cv2.line(img, center, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    cv2.line(img, center, tuple(imgpts[2].ravel()), (0, 0, 255), 5)

def draw_points(img, pts):
    for pt in pts:
        cv2.circle(img, (pt[0],pt[1]), 3, (0, 0, 255), -1)

# read manuel points
pts1 = pd.read_csv('../tmp/manual_pt1.csv', sep=',').values
pts2 = pd.read_csv('../tmp/manual_pt2.csv', sep=',').values

# Load previously saved data
K, dist = np.load(MAT_CAMERA), np.load(MAT_DIST_COEFF)

img1 = cv2.imread('../data/img1.jpg')


obj = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)

# project world coordinates to frame 1
r_vec_id, _ = cv2.Rodrigues(np.identity(3))
imgpts1, _ = cv2.projectPoints(obj, r_vec_id, np.zeros(3), K, dist)

# map to second image

# https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
# Normalize for Essential Matrix calaculation
pts_l_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1).astype(dtype=np.float32), cameraMatrix=K, distCoeffs=dist)
pts_r_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1).astype(dtype=np.float32), cameraMatrix=K, distCoeffs=dist)

E, _ = cv2.findEssentialMat(pts_l_norm, pts_r_norm, method=cv2.RANSAC, prob=0.999, threshold=0.1) # TODO test different settings

# recover relative camera rotation and translation from essential matrix and the corresponding points
points, R, t, _ = cv2.recoverPose(E, pts1, pts2)

# IMPORTANT: R and t may need to be inverted:
# see https://answers.opencv.org/question/31421/opencv-3-essentialmatrix-and-recoverpose/
# and https://stackoverflow.com/questions/54486906/correctly-interpreting-the-pose-rotation-and-translation-after-recoverpose-f
#R = np.linalg.inv(R)
#t = np.flip(t)

img2 = cv2.imread('../data/img2.jpg')

# project world coordinates to frame 2
r_vec, _ = cv2.Rodrigues(R)
imgpts2, _ = cv2.projectPoints(obj, r_vec, t, K, dist)


# show images

img1 = cv2.resize(img1, (960, 540))
draw_points(img1, pts1)
draw(img1, imgpts1)
cv2.imshow('img1', img1)

img2 = cv2.resize(img2, (960, 540))
draw_points(img2, pts2)
draw(img2, imgpts1)
cv2.imshow('img2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()