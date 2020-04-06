import numpy as np
import cv2

""" Contains project config
Paths for INPUT and OUTPUT video files
Paths for INTERMEDIATE files
"""

DEMO_RESIZE = (960, 540)

SCALING_FACTOR = 9
CALIBR_SQUARE_SIZE_MM = 25
CALIBR_BOARD_SHAPE = (8, 6)


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
CALIB_VIDEO_PATH = DATA_PATH + 'visual-geometry-calibration.MTS'

# Path to input video
VIDEO_PATH = DATA_PATH + 'visual-geometry-video.MTS'

# Path to output video
VIDEO_OUT_PATH = TMP_PATH + 'output.avi'
VIDEO_OUT_STEREO_PATH = TMP_PATH + 'output_stereo.avi'

# Intermediate files
CAMERA_MATRIX = TMP_PATH + 'cmatrix.npy'
CAMERA_DIST_COEFF = TMP_PATH + 'dist.npy'
MANUAL_MATCH_POINTS_1 = TMP_PATH + 'manual_pt1.csv'
MANUAL_MATCH_POINTS_2 = TMP_PATH + 'manual_pt2.csv'
MANUAL_MATCH_POINTS_3 = TMP_PATH + 'manual_pt3.csv'

K, dist = None, None


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