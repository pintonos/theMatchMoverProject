import cv2
import numpy as np
from matplotlib import pyplot as plt
from Constants import *
import pandas as pd


def draw_lines(im, center, obj_pts):
    for pt in obj_pts:
        cv2.line(im, center, pt, (255, 0, 0), 5)

def draw_points(img, pts):
    for pt in pts:
        cv2.circle(img, (pt[0],pt[1]), 3, (0, 0, 255), -1)

# transforms point according to formula p2 = R*p1 + t returns a tuple (x, y)
def transform_point(pt1, R, t):
    # from image to homogeneuos
    pt1_homo = np.append(pt1, 1).reshape(3,1)
    # map from img1 to img2
    pt_rot = np.matmul(R, pt1_homo)
    pt2_hom = np.add(pt_rot, t) 
    # back from homogeneous to image
    pt2 = pt2_hom / pt2_hom[2]
    # reshape to int tuple
    return (int(pt2[0][0]), int(pt2[1][0])) 


# read manuel points
pts1 = pd.read_csv('../tmp/manual_pt1.csv', sep=',').values
pts2 = pd.read_csv('../tmp/manual_pt2.csv', sep=',').values

img1 = cv2.imread('../data/img1.jpg')

center = (464, 302)
obj_pts1 = [(464,252), (514,302),(428,338)]

img1 = cv2.resize(img1, (960, 540))
draw_lines(img1, center, obj_pts1)
draw_points(img1, pts1)

cv2.imshow('img1', img1)
cv2.waitKey(0)

# map to second image

# Load K and dist from calibration
K, dist = np.load(MAT_CAMERA), np.load(MAT_DIST_COEFF)

# https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
# Normalize for Essential Matrix calaculation
pts_l_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1).astype(dtype=np.float32), cameraMatrix=K, distCoeffs=dist)
pts_r_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1).astype(dtype=np.float32), cameraMatrix=K, distCoeffs=dist)

E, _ = cv2.findEssentialMat(pts_l_norm, pts_r_norm, method=cv2.RANSAC, prob=0.999, threshold=0.1) # TODO test different settings

# recover relative camera rotation and translation from essential matrix and the corresponding points
_, R, t, _ = cv2.recoverPose(E, pts1, pts2)

# IMPORTANT: R and t may need to be inverted:
# see https://answers.opencv.org/question/31421/opencv-3-essentialmatrix-and-recoverpose/
# and https://stackoverflow.com/questions/54486906/correctly-interpreting-the-pose-rotation-and-translation-after-recoverpose-f
#R = np.linalg.inv(R)
#t = np.flip(t)

# transform points
obj_pts2 = []
for imagept in obj_pts1:
    obj_pts2.append(transform_point(imagept, R, t))
    
# transform center
center = transform_point(center, R, t)

print(obj_pts2)
print(center)

img2 = cv2.imread('../data/img2.jpg')
img2 = cv2.resize(img2, (960, 540))
draw_lines(img2, center, obj_pts2)
draw_points(img2, pts2)

cv2.imshow('img2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()