import cv2
import numpy as np
from Constants import *
from matcher import *

np.set_printoptions(suppress=True)

SKIP_FPS = 30
MAX_FPS = 80

OBJECT_POSITION = np.asarray(np.float32([1, 1.7, 27]))

# -60 degrees about x-axis
OBJECT_ORIENTATION = np.float32([ # https://www.andre-gaschler.com/rotationconverter/
    [1, 0, 0],
    [0, 0.5, 0.8660254],
    [0, -0.8660254, 0.5]
])

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
r_vec_id, _ = cv2.Rodrigues(OBJECT_ORIENTATION)
t_vec = OBJECT_POSITION
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

    pts1, pts2 = get_points(img1, img2, 'FAST', True, 'FLANN')

    E, _ = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, prob=0.999, threshold=1, cameraMatrix=K)

    # recover relative camera rotation and translation from essential matrix and the corresponding points
    points, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    # project world coordinates to frame 2
    t += np.expand_dims(t_vec, axis=1)  # add scaling factor
    R = R @ OBJECT_ORIENTATION
    r_vec, _ = cv2.Rodrigues(R, dst=dist)
    imgpts2, _ = cv2.projectPoints(obj, r_vec, t, K, dist)

    img2 = draw(img2, imgpts2)

    #cv2.imshow('img', img2)
    #cv2.waitKey(1)

    out.write(img2)


video_cap.release()
out.release()
cv2.destroyAllWindows()