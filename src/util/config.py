import numpy as np

""" 
Contains project config
Paths for INPUT and OUTPUT video files
Paths for INTERMEDIATE files
"""

# Data & Tmp path
DATA_PATH = '../resources/data/'

# Path to calibration video
CALIB_VIDEO_PATH = DATA_PATH + 'visual-geometry-calibration.MTS'

# Path to input video
VIDEO_PATH = DATA_PATH + 'visual-geometry-video3.MTS'

# Path to output video
VIDEO_OUT_PATH = DATA_PATH + 'output.avi'

# saved keyframe tracing files
KEYFRAMES_PATH = DATA_PATH + 'keyframes.npy'
KEYFRAMES_IDX_PATH = DATA_PATH + 'keyframe_idx.npy'

# Calibration files
CAMERA_MATRIX = DATA_PATH + 'cmatrix.npy'
CAMERA_DIST_COEFF = DATA_PATH + 'dist.npy'

# template CSV file path
REF_POINTS = DATA_PATH + 'reference_{frame}.csv'


def load_interm(camera_matrix, camera_dist_coeff):
    global K
    global dist

    try:
        K = np.load(camera_matrix)
        dist = np.load(camera_dist_coeff)
    except IOError:
        print('ERROR: No camera matrix or distortion coefficient initialized. Calibrate camera first.')
        exit(-1)


# init K and dist
K, dist = None, None
load_interm(CAMERA_MATRIX, CAMERA_DIST_COEFF)
