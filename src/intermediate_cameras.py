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


def find_trace_points(video, idx1, idx2):
    '''
    finds point matches that are preserved between idx1 and idx2
    :param video: video file with more than |idx2| frames
    :param idx1: index to start with
    :param idx2: index to end with
    :return: list of dict with structure [{'startPoint': 2DPoint, 'endPoint': 2DPoint}],
        startPoint is the 2DPoint of the first frame, endPoint is the 2DPoint of the last frame
    '''
    if idx2 - idx1 <= 0:
        print("warning, called find_trace_points with 0 or negative frame indexes")
        return []

    curr_idx = -1
    success = True
    traced_matches = None
    last_frame = None

    while success:
        success, frame = video.read()
        curr_idx += 1

        if curr_idx == idx1:
            # start tracing
            last_frame = frame

        if curr_idx > idx1:
            # trace
            match_points_1, match_points_2, matches = get_points(last_frame, frame)

            if traced_matches is None:
                traced_matches = [{
                    'startPoint': match_points_1[i],
                    'from': x.queryIdx,
                    'to': x.trainIdx,
                    'endPoint': match_points_2[i]} for i, x in enumerate(matches)]
            else:
                new_matches = dict([(x.queryIdx, x.trainIdx) for x in matches])
                for match in traced_matches:
                    new_from = match['to']
                    if new_from in new_matches:
                        match['from'] = new_from
                        match['to'] = new_matches[new_from]
                        match['endPoint'] = match_points_2[list(new_matches.keys()).index(new_from)]
                    else:
                        match['to'] = None

                traced_matches = list(filter(lambda m: m['to'] is None, traced_matches))

        if curr_idx > idx2:
            # end tracing
            break

    return traced_matches



video.release()
out.release()
cv2.destroyAllWindows()
