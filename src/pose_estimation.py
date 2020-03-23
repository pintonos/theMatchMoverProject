#!/usr/bin/python3
import cv2
import numpy as np
from src.Constants import *

#TODO https://stackoverflow.com/questions/22180923/how-to-place-object-in-video-with-opencv/22192565#22192565

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


MAX_FRAMES = 300

# Load previously saved data
mtx, dist = np.load(MTX), np.load(MAT_DIST_COEFF)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = getObjectPointsStructure()

axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                   [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

video_cap = cv2.VideoCapture(CALIB_VIDEO_PATH)
number_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('read', number_frames, 'frames ...')

# Get the Default resolutions
frame_width = int(video_cap.get(3))
frame_height = int(video_cap.get(4))

scale_percent = 60  # percent of original size
rescaled_width = int(frame_width * scale_percent / 100)
rescaled_height = int(frame_height * scale_percent / 100)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(VIDEO_OUT, fourcc, 20.0, (rescaled_width, rescaled_height))

success = 1
frame_counter = 0
while success and frame_counter < MAX_FRAMES:
    # function extract frames
    success, img = video_cap.read()
    frame_counter = frame_counter + 1
    print("Frame: {}/{}".format(frame_counter, MAX_FRAMES))

    # resize for faster testing
    dim = (rescaled_width, rescaled_height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img, corners2, imgpts)

    out.write(img)

video_cap.release()
out.release()
cv2.destroyAllWindows()
