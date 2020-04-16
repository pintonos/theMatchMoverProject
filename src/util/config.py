import numpy as np
import cv2
from enum import Enum

""" Contains project config
Paths for INPUT and OUTPUT video files
Paths for INTERMEDIATE files
"""

DEMO_RESIZE = (960, 540)

SCALING_FACTOR = 9
CALIBR_SQUARE_SIZE_MM = 25
CALIBR_BOARD_SHAPE = (8, 6)

INIT_POSITION = np.asarray([0, 0, 0], dtype=float)
INIT_ORIENTATION = np.identity(3)


def get_obj_point_structure():
    """Prepare object points dependant of BOARD_SIZE"""
    object_points = np.zeros((CALIBR_BOARD_SHAPE[0] * CALIBR_BOARD_SHAPE[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:CALIBR_BOARD_SHAPE[0], 0:CALIBR_BOARD_SHAPE[1]].T.reshape(-1, 2)
    return object_points


CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, CALIBR_SQUARE_SIZE_MM, 0.001)

# Data & Tmp path
DATA_PATH = '../resources/data/'
TMP_PATH = '../resources/tmp/'

# Path to calibration video
CALIB_VIDEO_PATH = DATA_PATH + 'visual-geometry-calibration2.MTS'

# Path to input video
VIDEO_PATH = DATA_PATH + 'visual-geometry-video2_1.MTS'

# Path to output video
VIDEO_OUT_PATH = TMP_PATH + 'output.avi'
VIDEO_OUT_STEREO_PATH = TMP_PATH + 'output_stereo.avi'

# Intermediate files
CAMERA_MATRIX = TMP_PATH + 'cmatrix2.npy'
CAMERA_DIST_COEFF = TMP_PATH + 'dist2.npy'
MANUAL_MATCH_POINTS_1 = TMP_PATH + 'manual_pt1.csv'
MANUAL_MATCH_POINTS_2 = TMP_PATH + 'manual_pt2.csv'
MANUAL_MATCH_POINTS_3 = TMP_PATH + 'manual_pt3.csv'
REF_POINTS_1 = TMP_PATH + 'reference_1.csv'
REF_POINTS_50 = TMP_PATH + 'reference_50.csv'
REF_POINTS_100 = TMP_PATH + 'reference_100.csv'
REF_POINTS_150 = TMP_PATH + 'reference_150.csv'

K, dist = None, None


class Detector(Enum):
    SIFT = 1
    SURF = 2
    FAST = 3
    ORB = 4


class Matcher(Enum):
    FLANN = 1
    BRUTE_FORCE = 2


def load_interm(camera_matrix, camera_dist_coeff):
    global K
    global dist

    try:
        K = np.load(camera_matrix)
        dist = np.load(camera_dist_coeff)
    except IOError:
        # do nothing
        pass


load_interm(CAMERA_MATRIX, CAMERA_DIST_COEFF)
load_interm('../' + CAMERA_MATRIX, '../' + CAMERA_DIST_COEFF)

if K is None or dist is None:
    print('No camera matrix or distortion coefficient initialized.')
