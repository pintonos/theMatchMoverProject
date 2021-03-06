from util import *

'''
Calibrate camera with calibration video
Will calculate camera matrix and distortion coefficient
'''

CALIBR_BOARD_SHAPE = (8, 6)
CALIBR_SQUARE_SIZE_MM = 25
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, CALIBR_SQUARE_SIZE_MM, 0.001)


def get_obj_point_structure():
    """
    Prepare object points dependant of BOARD_SIZE
    """
    object_points = np.zeros((CALIBR_BOARD_SHAPE[0] * CALIBR_BOARD_SHAPE[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:CALIBR_BOARD_SHAPE[0], 0:CALIBR_BOARD_SHAPE[1]].T.reshape(-1, 2)
    return object_points


SAVE = False  # Save results to disk
N_TH_FRAME_TO_USE = 3  # use only every n-th frame for calibration

point_structure = get_obj_point_structure()

obj_points_3d = []  # 3d point in real world space
img_points_2d = []  # 2d points in image plane.

video = cv2.VideoCapture(CALIB_VIDEO_PATH)
frames_total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

logging.info('Processing {} frames ...'.format(frames_total))

count = 0
success = True
demo_img = None

while success:
    success, img = video.read()

    # Store demo image from middle frame
    if count == frames_total // 2:
        demo_img = img
    count += 1

    # Use every n-th element for calibration
    if not success or count % N_TH_FRAME_TO_USE != 0:
        continue

    # Find chessboard corners
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CALIBR_BOARD_SHAPE, None)

    # For identified corners, add object points and image points
    if ret:
        obj_points_3d.append(point_structure)

        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        img_points_2d.append(corners_refined)

        # DEBUG: Draw and display corners
        # cv2.imshow('Corners', cv2.drawChessboardCorners(img, CALIBR_BOARD_SHAPE, corners_refined, ret))
        # cv2.waitKey(50)

video.release()
cv2.destroyAllWindows()

logging.info('Video processing completed, calculating calibration matrix...')

# Calibrate camera
gray = cv2.cvtColor(demo_img, cv2.COLOR_BGR2GRAY)
logging.debug(obj_points_3d)
logging.debug(img_points_2d)

ret, mtx, dist_tmp, rvecs, tvecs = cv2.calibrateCamera(obj_points_3d, img_points_2d, gray.shape[::-1], None, None)

h, w = demo_img.shape[:2]
camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist_tmp, (w, h), 1, (w, h))

# Save obtained camera matrix and distance coefficient
if SAVE:
    np.save(CAMERA_DIST_COEFF, dist_tmp)
    np.save(CAMERA_MATRIX, camera_mtx)
logging.info('Camera matrix: {}'.format(str(camera_mtx)))

# Calculate mean error
mean_error = 0
for i in range(len(obj_points_3d)):
    tmp_proj_points, _ = cv2.projectPoints(obj_points_3d[i], rvecs[i], tvecs[i], mtx, dist_tmp)
    error = cv2.norm(img_points_2d[i], tmp_proj_points, cv2.NORM_L2) / len(tmp_proj_points)
    mean_error += error

logging.info('Total error: {}'.format(mean_error / len(obj_points_3d)))
