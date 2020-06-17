from functions import *
from util import models
from matplotlib import pyplot as plt
import cv2

""" 
Functions required for automatic point matching

These functions are used for point matching between to frames
in order to perform a stereo calibration.
"""


def lowes_ratio_test(kp1, kp2, matches, threshold=0.8):
    """
    Ratio test as per Lowe's paper.
    :param threshold: threshold to compare matches
    :param kp1: keypoints image 1
    :param kp2: keypoints image 2
    :param matches: matches between images
    :return: good matches
    """
    pts1 = []
    pts2 = []
    good = []
    for i, match in enumerate(matches):
        if len(match) < 2:
            continue
        (m, n) = match
        if m.distance < threshold * n.distance:  # TODO tweak ratio
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            good.append(m)

    return pts1, pts2, good


def get_flann_matches(kp1, des1, kp2, des2, detector):
    index_params = None

    # FLANN parameters
    if detector == Detector.ORB:
        index_params = dict(algorithm=6,
                            table_number=12,
                            key_size=20,
                            multi_probe_level=2)
    else:
        index_params = dict(algorithm=0, trees=5)

    if index_params is None:
        raise Exception('Unknown detector [' + str(detector) + ']')

    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Filter for good matches
    pts1, pts2, matches = lowes_ratio_test(kp1, kp2, matches)

    return np.int32(pts1), np.int32(pts2), matches


def get_brute_force_matches(kp1, des1, kp2, des2, detector, ratio_test=True):

    if detector == models.Detector.ORB:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=not ratio_test)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=not ratio_test)

    if ratio_test:
        matches = bf.knnMatch(des1, des2, k=2)
        pts1, pts2, good_matches = lowes_ratio_test(kp1, kp2, matches)
    if not ratio_test:
        # Match descriptors
        matches = bf.match(des1, des2)

        pts1, pts2 = [], []
        # sort matches and drop worst ones
        matches = sorted(matches, key=lambda x: x.distance)
        num_good_matches = len(matches) // 2

        matches = matches[:num_good_matches]
        for i, match in enumerate(matches):
            pts2.append(kp2[match.trainIdx].pt)
            pts1.append(kp1[match.queryIdx].pt)

        good_matches = matches[:num_good_matches]

    return np.int32(pts1), np.int32(pts2), good_matches


def get_points(img1, img2, detector=models.Detector.ORB, filtered=True, matcher=models.Matcher.BRUTE_FORCE, show_matches=False):

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Filter images to remove noise like a gaussian, but preserves edges
    if filtered:
        gray1 = cv2.bilateralFilter(gray1, 5, 50, 50)
        gray2 = cv2.bilateralFilter(gray2, 5, 50, 50)

    # Find the keypoints and descriptors
    kp1, kp2 = None, None
    des1, des2 = None, None
    if detector == models.Detector.SIFT:
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
    elif detector == models.Detector.SURF:
        surf = cv2.xfeatures2d.SURF_create()
        kp1, des1 = surf.detectAndCompute(gray1, None)
        kp2, des2 = surf.detectAndCompute(gray2, None)
    elif detector == models.Detector.FAST:
        fast = cv2.FastFeatureDetector_create()
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.compute(gray1, fast.detect(gray1, None))
        kp2, des2 = sift.compute(gray2, fast.detect(gray2, None))
    elif detector == models.Detector.ORB:
        orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.5)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

    if kp1 is None or kp2 is None or des1 is None or des2 is None:
        raise Exception('Unknown detector [' + str(detector) + ']')

    # Match points
    pts1, pts2, matches = None, None, None
    if matcher == models.Matcher.FLANN:
        pts1, pts2, matches = get_flann_matches(kp1, des1, kp2, des2, detector)
    elif matcher == models.Matcher.BRUTE_FORCE:
        pts1, pts2, matches = get_brute_force_matches(kp1, des1, kp2, des2, detector)

    if pts1 is None or pts2 is None:
        raise Exception('Unknown matcher [' + str(matcher) + ']')

    # Subpixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    pts1 = cv2.cornerSubPix(gray1, np.float32(pts1), (5, 5), (-1, -1), criteria)
    pts2 = cv2.cornerSubPix(gray2, np.float32(pts2), (5, 5), (-1, -1), criteria)

    if show_matches:
        matches = sorted(matches, key=lambda x: x.distance)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:100], None, flags=2)
        plt.imshow(img3), plt.show()
    
    return pts1, pts2, matches
