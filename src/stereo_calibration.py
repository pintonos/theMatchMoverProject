from functions import *
from util import *

np.set_printoptions(suppress=True)

# Check intermediate values
if K is None or dist is None:
    raise Exception('Camera matrix or distortion coefficient not found')

SKIP_FPS = 30
MAX_FPS = 80

# Chooses the frame to match points
# 0: always first frame
# n > 0: use the n-th predecessor
COMPARE_FRAME = 15

# Points for a 3D cube
img_points_3d = get_3d_cube_points()

video = cv2.VideoCapture(VIDEO_PATH)
frames_total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print('Processing', frames_total, 'frames ...')

# Get the Default resolutions
frame_width = int(video.get(3))
frame_height = int(video.get(4))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(VIDEO_OUT_STEREO_PATH, fourcc, 20.0, (frame_width, frame_height))

# Project world coordinates to first frame
r_vec_id, _ = cv2.Rodrigues(OBJECT_ORIENTATION)
t_vec = OBJECT_POSITION
proj_points_2d, _ = cv2.projectPoints(img_points_3d, r_vec_id, t_vec, K, dist)

success, first_frame = video.read()
first_frame = draw(first_frame, proj_points_2d)

# dst1 = get_harris_corner(first_frame)  # TODO what's this for?
out.write(first_frame)

# Project coordinates to every following frame
count = 0
frame = 0

compare_frames = [first_frame]
while success and count < MAX_FPS:
    count += 1
    success, frame = video.read()
    print("Frame: {}/{}".format(count, frames_total))

    if count < SKIP_FPS:
        continue

    if COMPARE_FRAME > 0:
        compare_frames = compare_frames[-COMPARE_FRAME:]

    # Automatic point matching
    match_points_1, match_points_2 = get_points(compare_frames[0], frame, Detector.FAST, True, Matcher.FLANN, showMatches=True)

    E, _ = cv2.findEssentialMat(match_points_1, match_points_2, method=cv2.RANSAC, prob=0.999, threshold=1, cameraMatrix=K)

    # Recover relative camera rotation and translation from essential matrix and the corresponding points
    _, R, t, _ = cv2.recoverPose(E, match_points_1, match_points_2, K)

    # Project world coordinates to frame
    t += np.expand_dims(t_vec, axis=1)  # add scaling factor
    R = R @ OBJECT_ORIENTATION
    r_vec, _ = cv2.Rodrigues(R, dst=dist)
    proj_points_img_2, _ = cv2.projectPoints(img_points_3d, r_vec, t, K, dist)

    compare_frames.append(frame)
    frame = draw(frame, proj_points_img_2)

    # DEBUG: Plot frame
    cv2.imshow('current_frame', frame)
    cv2.waitKey(1)
    out.write(frame)

video.release()
out.release()
cv2.destroyAllWindows()
