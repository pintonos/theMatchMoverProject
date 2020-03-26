import cv2
import numpy as np
from matplotlib import pyplot as plt
from Constants import *

video_cap = cv2.VideoCapture(VIDEO_PATH)
number_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('read', number_frames, 'frames ...')

frame_counter = 0
success = True
img1, img2 = 0, 0
while success:
    if frame_counter == 0:
        success, img1 = video_cap.read()
        frame_counter = frame_counter + 1
        continue

    success, img1 = video_cap.read()
    frame_counter = frame_counter + 1
    print("Frame: {}/{}".format(frame_counter, number_frames))

    success, img2 = video_cap.read()

    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

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

    # get Fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    print("F:")
    print(F)

    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Load previously saved data
    K, _ = np.load(MAT_CAMERA), np.load(MAT_DIST_COEFF)

    # get E from equation: E = K'^T * F * K
    E = np.multiply(np.multiply(np.transpose(K), F), K)

    print("E:")
    print(E)

    # recover relative camera rotation and translation from essential matrix and the corresponding points
    inlier_points, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    
    print(inlier_points, 'inliers points found')
    print("R:")
    print(R)
    print("t:")
    print(t, '\n')

    success, img1 = video_cap.read()


