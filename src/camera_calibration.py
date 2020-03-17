import numpy as np
import cv2
import os

VIDEO_PATH = '../data/visual-geometry-calibration.MTS'

# checkerboard characteristics
SQUARE_SIZE = 25
BOARD_SIZE = (8, 6)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, SQUARE_SIZE, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

video_cap = cv2.VideoCapture(VIDEO_PATH)
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

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, BOARD_SIZE, corners2, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(50)

print('reading in complete, calculating matrix...')
cv2.destroyAllWindows()

gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
# calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

np.save(os.path.join('..', 'tmp', 'matrix.npy'), mtx)
np.save(os.path.join('..', 'tmp', 'dist.npy'), dist)

h, w = test_img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
# dst = cv2.undistort(test_img, mtx, dist, None, newcameramtx)
# x, y, w, h = roi
# dst = dst[y:y + h, x:x + w]
# cv2.imshow("Orignal", test_img)
# cv2.imshow("Undistorted", dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total error: ", mean_error / len(objpoints))
print(newcameramtx)

#np.save(os.path.join('..', 'tmp', 'cmatrix.npy'), newcameramtx)
