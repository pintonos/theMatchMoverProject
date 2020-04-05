import cv2
import numpy as np
from matplotlib import pyplot as plt
from src.Constants import *
import pandas as pd

np.set_printoptions(suppress=True)

OBJECT_POSITION = np.asarray(np.float32([1, 1.7, 27]))

# -60 degrees about x-axis
OBJECT_ORIENTATION = np.float32([ # https://www.andre-gaschler.com/rotationconverter/
    [1, 0, 0],
    [0, 0.5, 0.8660254],
    [0, -0.8660254, 0.5]
])


def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


def draw_points(img, pts):
    for i, pt in enumerate(pts):
        cv2.circle(img, (pt[0], pt[1]), 3, (0, 0, 255), -1)
        cv2.putText(img, str(i), (pt[0], pt[1]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
        # print(i, " ", pt//2)


def invert(R, t):
    backRotation = np.c_[R, t]
    backRotation = np.append(backRotation, [[0, 0, 0, 1]], axis=0)
    # https://answers.opencv.org/question/31421/opencv-3-essentialmatrix-and-recoverpose/
    backRotation = np.linalg.inv(backRotation)

    R_inv = backRotation[np.ix_([0, 1, 2], [0, 1, 2])]
    t_inv = backRotation[:, 3][:-1].reshape(3, 1)
    return R_inv, t_inv


def get_points(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    return pts1, pts2


def stereo_view_map(pts1, pts2, K):
    # https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
    # Normalize for Essential Matrix calculation
    # pts_l_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1).astype(dtype=np.float32), cameraMatrix=K, distCoeffs=dist)
    # pts_r_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1).astype(dtype=np.float32), cameraMatrix=K, distCoeffs=dist)

    E, _ = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, prob=0.999, threshold=1,
                                cameraMatrix=K)  # TODO test different settings

    # E_inv = np.linalg.inv(E)
    # print(np.around(E @ np.linalg.inv(E)).astype(int))

    # p1 = np.float32([[600, 315, 1]]).reshape(3, 1)
    # print(E_inv @ p1)

    # recover relative camera rotation and translation from essential matrix and the corresponding points
    points, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    # project world coordinates to frame 2
    t = np.add(t, np.expand_dims(t_vec, axis=1))  # add scaling factor
    R = R @ OBJECT_ORIENTATION
    r_vec, _ = cv2.Rodrigues(R, dst=dist)
    imgpts2, _ = cv2.projectPoints(obj, r_vec, t, K, dist)

    return imgpts2


# Load previously saved data
K, dist = np.load(MAT_CAMERA), np.load(MAT_DIST_COEFF)

obj = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]]) * 1.5

# project world coordinates to frame 1
r_vec_id, _ = cv2.Rodrigues(OBJECT_ORIENTATION)

t_vec = OBJECT_POSITION
imgpts1, _ = cv2.projectPoints(obj, r_vec_id, t_vec, K, dist)

img1 = cv2.imread('../data/img1.jpg')
img2 = cv2.imread('../data/img2.jpg')
img3 = cv2.imread('../data/img3.jpg')

# map images
pts1, pts2 = get_points(img1, img2)
pts1_3, pts3 = get_points(img2, img3)

imgpts2 = stereo_view_map(pts1, pts2, K)
imgpts3 = stereo_view_map(pts1_3, pts3, K)

# show images
draw_points(img1, pts1)
draw(img1, imgpts1)
img1 = cv2.resize(img1, (960, 540))
cv2.imshow('img1', img1)

draw_points(img2, pts2)
draw(img2, imgpts2)
img2 = cv2.resize(img2, (960, 540))
cv2.imshow('img2', img2)

draw_points(img3, pts3)
draw(img3, imgpts3)
img3 = cv2.resize(img3, (960, 540))
cv2.imshow('img3', img3)

cv2.waitKey(0)
cv2.destroyAllWindows()
