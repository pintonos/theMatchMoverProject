import cv2
import numpy as np
from Constants import *

SKIP_FPS = 30
MAX_FPS = 195
SCALING_FACTOR = 9

def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


def get_harris_corner(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    return dst


def get_points(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    return pts1, pts2

# Load previously saved data
K, dist = np.load(MAT_CAMERA), np.load(MAT_DIST_COEFF)

obj = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]])

video_cap = cv2.VideoCapture(VIDEO_PATH)
number_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('read', number_frames, 'frames ...')

# Get the Default resolutions
frame_width = int(video_cap.get(3))
frame_height = int(video_cap.get(4))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(VIDEO_OUT_STEREO, fourcc, 20.0, (frame_width, frame_height))

# project world coordinates to frame 1
r_vec_id, _ = cv2.Rodrigues(np.identity(3))
t_vec = np.float32(np.asarray([0, 0, SCALING_FACTOR]))
imgpts1, _ = cv2.projectPoints(obj, r_vec_id, t_vec, K, dist)

success, img1 = video_cap.read()
dst1 = get_harris_corner(img1)

out.write(img1)

frame_counter = 0
success = True
img2 = 0
while success and frame_counter < MAX_FPS:
    frame_counter = frame_counter + 1
    success, img2 = video_cap.read()
    print("Frame: {}/{}".format(frame_counter, number_frames))

    if frame_counter < SKIP_FPS:
        continue

    pts1, pts2 = get_points(img1, img2)
    print(pts1, pts2)

    E, _ = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, prob=0.999, threshold=1, cameraMatrix=K)

    # recover relative camera rotation and translation from essential matrix and the corresponding points
    points, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    # project world coordinates to frame 2
    t += np.expand_dims(t_vec, axis=1)  # add scaling factor
    r_vec, _ = cv2.Rodrigues(R, dst=dist)
    imgpts2, _ = cv2.projectPoints(obj, r_vec, t, K, dist)

    img2 = draw(img2, imgpts2)

    cv2.imshow('img', img2)
    cv2.waitKey(1)

    out.write(img2)


video_cap.release()
out.release()
cv2.destroyAllWindows()