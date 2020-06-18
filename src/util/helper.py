from util import *
from functions.keyframes import get_all_keyframes
from functions.stereo_functions import get_F
import os
import cv2
import numpy as np


def get_video_streams():
    """
    Get reader and writer handlers of video.
    """
    video = cv2.VideoCapture(VIDEO_PATH)

    # Get the Default resolutions
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, 20.0, (frame_width, frame_height))
    return video, out


def get_frame(frame_index):
    """
    Get frame of video at given index.
    """
    reader = cv2.VideoCapture(VIDEO_PATH)
    img = None
    for i in range(frame_index + 1):
        _, img = reader.read()
    return img


def get_inlier_points(points_3d, points_2d, inliers):
    """
    Filter out lists of 2D and 3D points coordinates according to a given list of inliers.
    """
    filtered_3d = []
    filtered_2d = []

    points_2d = np.swapaxes(points_2d, 0, 1)
    for i in range(len(inliers)):
        in_index = inliers[i][0]
        filtered_2d.append(points_2d[in_index])
        filtered_3d.append(points_3d[in_index])

    filtered_2d = np.swapaxes(np.asarray(filtered_2d), 0, 1)
    return np.asarray(filtered_3d), filtered_2d


def correct_matches(points, halfway_idx):
    """
    Get corrected 2D point coordinates of two keyframes using the optimal triangulation method.
    """
    if halfway_idx >= len(points):
        return None, None

    F, _ = get_F(points[0], points[-1])

    pts1 = np.reshape(points[0], (1, len(points[0]), 2))
    pts2 = np.reshape(points[halfway_idx], (1, len(points[halfway_idx]), 2))
    pts1, pts2 = cv2.correctMatches(F, pts1, pts2)

    return pts1[0], pts2[0]


def get_3d_points_for_consecutive_frames(points_3d, points_2d):
    """
    Get a list of world coordinates for given 2D points. Assumption: all intermediate frames see the same world points.
    """
    points_3d = [points_3d for _ in range(points_2d.shape[0])]
    return np.asarray(points_3d)


def load_keyframes(start, end):
    # get keyframes if they are saved on disk
    if os.path.isfile(KEYFRAMES_IDX_PATH):  # load from disk
        keyframes = np.load(KEYFRAMES_PATH, allow_pickle=True)
        keyframe_idx = np.load(KEYFRAMES_IDX_PATH, allow_pickle=True)
    else:  # find new keyframes
        keyframes, keyframe_idx = get_all_keyframes(start, end)

        # save data
        np.save(KEYFRAMES_PATH, keyframes)
        np.save(KEYFRAMES_IDX_PATH, keyframe_idx)

    return keyframes, keyframe_idx
