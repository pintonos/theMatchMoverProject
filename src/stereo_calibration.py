from functions import *
from util import *

# Check intermediate values
if K is None or dist is None:
    raise Exception('Camera matrix or distortion coefficient not found')

MAX_FPS = 100
SKIP_FPS = 30

# Points for a 3D cube
img_points_3d = get_3d_axis(,

video = cv2.VideoCapture(VIDEO_PATH)
frames_total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print('Processing', frames_total, 'frames ...')

# Get the Default resolutions
frame_width = int(video.get(3))
frame_height = int(video.get(4))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(VIDEO_OUT_STEREO_PATH, fourcc, 20.0, (frame_width, frame_height))

R = INIT_ORIENTATION
t_vec = INIT_POSITION

success, first_frame = video.read()

# Project coordinates to every following frame
count = 0
frame = 0

while success and count < MAX_FPS:
    count += 1
    success, frame = video.read()
    print("Frame: {}/{}".format(count, frames_total))

    if count < SKIP_FPS:
        continue

    # Automatic point matching
    match_points_1, match_points_2 = get_points(first_frame, frame)

    R, t_vec = get_R_and_t(match_points_1, match_points_2, K)
    r_vec, _ = cv2.Rodrigues(R, dst=dist)
    img_points_2d, _ = cv2.projectPoints(img_points_3d, r_vec, t_vec, K, dist)

    draw_points(frame, match_points_2)
    plot_show_img(frame, img_points_2d, 'img_1', axis=True)

    out.write(frame)

    # cv2.imshow('current_frame', frame)
    cv2.waitKey(1)

video.release()
out.release()
cv2.destroyAllWindows()
