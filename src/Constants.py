# checkerboard characteristics
import numpy as np
import cv2

SQUARE_SIZE = 25
BOARD_SIZE = (8, 6)

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, SQUARE_SIZE, 0.001)

# calibration video
CALIB_VIDEO_PATH = '../data/visual-geometry-calibration.MTS'

# video
VIDEO_PATH = '../data/visual-geometry-video.MTS'

# images
IMAGE_PATH = '../data/'

# tmp files
VIDEO_OUT = '../tmp/output.avi'
VIDEO_OUT_STEREO = '../tmp/output_stereo.avi'
MAT_CAMERA = '../tmp/cmatrix.npy'
MTX = '../tmp/matrix.npy'
MAT_DIST_COEFF = '../tmp/dist.npy'


def getObjectPointsStructure():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    return objp
