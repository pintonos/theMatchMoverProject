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
CONFIG_PATH = 'config.txt'

# template CSV file path
REF_POINTS = DATA_PATH + 'reference_{frame}.csv'

# define global variables
global K
global dist
global CALIB_VIDEO_PATH
global VIDEO_PATH
global VIDEO_OUT_PATH

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
except IOError:
    print('ERROR: Config.txt is missing or invalid.')
    exit(-1)



