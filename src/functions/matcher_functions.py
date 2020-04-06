import cv2
import numpy as np

def get_harris_corner(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    return dst

def lowes_ratio_test(kp1, kp2, matches):
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, match in enumerate(matches):
        if len(match) < 2:
            continue
        (m, n) = match
        if m.distance < 0.8 * n.distance: # TODO tweak ratio
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    
    return pts1, pts2


def get_flann_matches(kp1, des1, kp2, des2, detector):
    # FLANN parameters
    if detector == 'SIFT':
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    elif detector == 'FAST':
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=12, # TODO play around
                        key_size=20,  
                        multi_probe_level=2)  

    search_params = dict(checks=50) # TODO play around
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # filter for good matches
    pts1, pts2 = lowes_ratio_test(kp1, kp2, matches)

    return np.int32(pts1), np.int32(pts2)


def get_brute_force_matches(kp1, des1, kp2, des2):
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, # NORM_L2 is good for SIFT/SURF, use NORM_HAMMING for ORB/BRIEF
        crossCheck=True) # crossCheck is an alternative to Lowe's ratio test

    # Match descriptors.
    matches = bf.match(des1, des2)

    pts1, pts2 = [], []

    matches = sorted(matches, key = lambda x:x.distance)
    num_good_matches = len(matches) // 2 # TODO tweak ratio

    matches = matches[:num_good_matches] # use only first half, 
    for i, match in enumerate(matches):
        pts2.append(kp2[match.trainIdx].pt)
        pts1.append(kp1[match.queryIdx].pt)

    return np.int32(pts1), np.int32(pts2)

def get_points(img1, img2, detector='SIFT', filter=True, matcher='FLANN'):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # filter images to remove noise
    # like a gaussian, but preserves edges
    if filter:
        gray1 = cv2.bilateralFilter(gray1, 5, 50, 50)
        gray2 = cv2.bilateralFilter(img2, 5, 50, 50)  

    # find the keypoints and descriptors
    if detector == 'SIFT':
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
    elif detector == 'FAST':
        fast = cv2.FastFeatureDetector_create()
        br = cv2.BRISK_create()
        kp1, des1 = br.compute(gray1,  fast.detect(gray1, None))
        kp2, des2 = br.compute(gray2,  fast.detect(gray2, None))

    if matcher == 'FLANN':
        pts1, pts2 = get_flann_matches(kp1, des1, kp2, des2, detector)
    elif matcher == 'BRUTE_FORCE':
        pts1, pts2 = get_brute_force_matches(kp1, des1, kp2, des2)

    return pts1, pts2