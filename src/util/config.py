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

# Path to calibration video
CALIB_VIDEO_PATH = DATA_PATH + 'visual-geometry-calibration.MTS'

# Path to input video
VIDEO_PATH = DATA_PATH + 'visual-geometry-video2_1.MTS'

# Path to output video
VIDEO_OUT_PATH = DATA_PATH + 'output.avi'
VIDEO_OUT_STEREO_PATH = DATA_PATH + 'output_stereo.avi'

# Intermediate files
CAMERA_MATRIX = DATA_PATH + 'cmatrix.npy'
CAMERA_DIST_COEFF = DATA_PATH + 'dist.npy'
REF_POINTS_0 = DATA_PATH + 'reference_0.csv'
REF_POINTS_10 = DATA_PATH + 'reference_10.csv'
REF_POINTS_18 = DATA_PATH + 'reference_18.csv'
REF_POINTS_34 = DATA_PATH + 'reference_34.csv'
REF_POINTS_100 = DATA_PATH + 'reference_100.csv'
REF_POINTS_117 = DATA_PATH + 'reference_117.csv'


K, dist = None, None


class Detector(Enum):
    SIFT = 1
    SURF = 2
    FAST = 3
    ORB = 4


class Matcher(Enum):
    FLANN = 1
    BRUTE_FORCE = 2


class Camera:
    def __init__(self, R, t):
        if R.shape[1] == 1:
            self.R, _ = cv2.Rodrigues(R)  # transform to rotation matrix
        else:
            self.R = R

        self.t = t

    def get_camera_matrix(self):
        return np.c_[self.R, self.t]

    def __str__(self):
        return str(np.c_[self.R, self.t])


def get_video_streams():
    video = cv2.VideoCapture(VIDEO_PATH)

    # Get the Default resolutions
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(VIDEO_OUT_STEREO_PATH, fourcc, 20.0, (frame_width, frame_height))
    return video, out


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
