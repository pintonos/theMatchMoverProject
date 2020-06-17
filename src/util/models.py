import cv2
from enum import Enum

"""
Helper Enums and Classes
"""


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
            self.R_vec = R
            self.R_mat, _ = cv2.Rodrigues(R)  # transform to rotation matrix
        else:
            self.R_vec, _ = cv2.Rodrigues(R)
            self.R_mat = R

        self.t = t
