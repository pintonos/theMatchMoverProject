from functions import *
import cv2
import pandas as pd

video = cv2.VideoCapture(VIDEO_PATH)
frames_total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print('Processing', frames_total, 'frames ...')

# Get the Default resolutions
frame_width = int(video.get(3))
frame_height = int(video.get(4))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, 20.0, (frame_width, frame_height))

pts = pd.read_csv(REF_POINTS_0, sep=',', header=None, dtype=float).values

success, key_frame_1 = video.read()
intermediate_frame = key_frame_1

count = 0
dst = pts[:4]  # axis points from first frame
R1, t1 = INIT_ORIENTATION, INIT_POSITION
cameras = [np.c_[R1, t1]]

all_matches = []
while success and count < 100:
    count += 1
    success, frame = video.read()
    print("Frame: {}/{}".format(count, frames_total))

    match_points_1, match_points_2, matches = get_points(intermediate_frame, frame)
    print(matches)
    #all_matches.append() # TODO

    intermediate_frame = frame
    break


def find_trace_points(video, id1, id2):

    return traced_matches



video.release()
out.release()
cv2.destroyAllWindows()
