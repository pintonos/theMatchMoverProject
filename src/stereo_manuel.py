import cv2
import numpy as np
from matplotlib import pyplot as plt
from Constants import *
import pandas as pd
 

# read manuel points
pts1 = pd.read_csv('../tmp/manual_pt1.csv', sep=',').values
pts2 = pd.read_csv('../tmp/manual_pt2.csv', sep=',').values

# get Fundamental matrix
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
print("F:\n", F)

# Load previously saved data
K, dist = np.load(MAT_CAMERA), np.load(MAT_DIST_COEFF)

# get E from equation: E = K'^T * F * K
E = np.matmul(np.matmul(np.transpose(K), F), K)

print("E:\n", E)

# recover relative camera rotation and translation from essential matrix and the corresponding points
inlier_points, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

# IMPORTANT: R and t may need to be inverted:
# see https://answers.opencv.org/question/31421/opencv-3-essentialmatrix-and-recoverpose/
# and https://stackoverflow.com/questions/54486906/correctly-interpreting-the-pose-rotation-and-translation-after-recoverpose-f

print("R:\n", R)
print("t:\n", t)


# TODO insert wireframe in img1 and map them to img2


# points can be mapped from the first image to the second by the formula: p2 = R*p1 + t
# see opencv link above
