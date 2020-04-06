import numpy as np
import cv2

""" Contains project config
Paths for INPUT and OUTPUT video files
Paths for INTERMEDIATE files
"""

CALIBR_SQUARE_SIZE_MM = 25
CALIBR_BOARD_SHAPE = (8, 6)


def get_obj_point_structure():
    """Prepare object points dependant of BOARD_SIZE"""
    object_points = np.zeros((CALIBR_BOARD_SHAPE[0] * CALIBR_BOARD_SHAPE[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:CALIBR_BOARD_SHAPE[0], 0:CALIBR_BOARD_SHAPE[1]].T.reshape(-1, 2)
    return object_points


CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, CALIBR_SQUARE_SIZE_MM, 0.001)

# Data & TMP path
DATA_PATH = '../data/'
TMP_PATH = '../tmp/'

# Path to calibration video
CALIB_VIDEO_PATH = DATA_PATH + 'visual-geometry-calibration.MTS'

# Path to input video
VIDEO_PATH = DATA_PATH + 'data/visual-geometry-video.MTS'

# Path to output video
VIDEO_OUT_PATH = TMP_PATH + 'output.avi'

# Intermediate files
CAMERA_MATRIX = TMP_PATH + 'cmatrix.npy'
CAMERA_DIST_COEFF = TMP_PATH + 'dist.npy'
