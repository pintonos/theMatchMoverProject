import os
import wget
import sys
import numpy as np

""" 
Contains project config
Paths for INPUT and OUTPUT video files
Paths for INTERMEDIATE files
"""

# Temporary file paths
DATA_PATH = '../resources/data/'
KEYFRAMES_PATH = DATA_PATH + 'keyframes.npy'
KEYFRAMES_IDX_PATH = DATA_PATH + 'keyframe_idx.npy'
CAMERA_MATRIX = DATA_PATH + 'cmatrix.npy'
CAMERA_DIST_COEFF = DATA_PATH + 'dist.npy'
CONFIG_PATH = 'default.conf'

# template CSV file path
REF_POINTS = DATA_PATH + 'reference_{frame}.csv'

# define global variables
global K
global dist
global CALIB_VIDEO_PATH
global VIDEO_PATH
global VIDEO_OUT_PATH
global SHOW_FRAMES

try:
    K = np.load(CAMERA_MATRIX)
    dist = np.load(CAMERA_DIST_COEFF)
except IOError:
    print('ERROR: No camera matrix or distortion coefficient initialized. Calibrate camera first.')
    exit(-1)

try:
    with open(CONFIG_PATH) as config_file:
        lines = config_file.read().splitlines()
        args = {}
        for line in lines:
            name, var = line.partition("=")[::2]
            args[name.strip()] = var

        CALIB_VIDEO_PATH = args['calib_video_path']
        VIDEO_PATH = args['input_video_path']
        VIDEO_OUT_PATH = args['output_video_path']

        try:
            SHOW_FRAMES = args['show_frames']
        except:
            SHOW_FRAMES = False

        try:
            if not os.path.exists(CALIB_VIDEO_PATH):
                CALIB_VIDEO_URL = args['calib_video_url']
                print('WARN: calibration video missing, downloading...')
                wget.download(CALIB_VIDEO_URL, CALIB_VIDEO_PATH)
        except:
            print('ERROR: no calibration video found, auto download failed')
            sys.exit(-1)

        try:
            if not os.path.exists(VIDEO_PATH):
                VIDEO_URL = args['input_video_url']
                print('WARN: input video missing, downloading...')
                wget.download(VIDEO_URL, VIDEO_PATH)
        except:
            print('ERROR: no input video found, auto download failed')
            sys.exit(-1)


except IOError:
    print('ERROR: default.conf is missing or invalid.')
    sys.exit(-1)
