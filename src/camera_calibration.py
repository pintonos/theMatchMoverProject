import numpy as np
import cv2
import os
from src.Constants import *


objp = getObjectPointsStructure()

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

video_cap = cv2.VideoCapture(CALIB_VIDEO_PATH)
number_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('read', number_frames, 'frames ...')

success = 1
count = 0
test_img = None
while success:

    # function extract frames 
    success, img = video_cap.read()
    count += 1

    # get test image from middle of data
    if count == number_frames // 2:
        test_img = img

    # use every 10th element for calibration
    if not success or count % 3 != 0:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        imgpoints.append(corners2)

        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, BOARD_SIZE, corners2, ret)
        # cv2.imshow('img', img)
        # cv2.imwrite(os.path.join('cal-sample_' + str(count) + '.jpg'), img)
        # cv2.waitKey(50)

video_cap.release()
print('reading in complete, calculating matrix...')
cv2.destroyAllWindows()

gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
# calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

np.save(MTX, mtx)
np.save(MAT_DIST_COEFF, dist)

h, w = test_img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total error: ", mean_error / len(objpoints))
print(newcameramtx)

#np.save(MAT_CAMERA, newcameramtx)
