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

success, first_frame = video.read()

count = 0
dst = pts[:4]  # axis points from first frame
while success and count < 100:
    count += 1
    success, frame = video.read()
    print("Frame: {}/{}".format(count, frames_total))

    # Automatic point matching
    match_points_1, match_points_2 = get_points(first_frame, frame)

    M, mask = cv2.findHomography(match_points_1, match_points_2, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    dst = cv2.perspectiveTransform(dst.reshape(-1,1,2), M)

    draw_axis(frame, dst)
    out.write(frame)

    first_frame = frame


    img = cv2.resize(frame, DEMO_RESIZE)
    cv2.imshow('frame', img)
    cv2.waitKey(1)

video.release()
out.release()
cv2.destroyAllWindows()