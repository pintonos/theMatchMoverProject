import functools
import operator

from functions import *
from util import *
import cv2
import pandas as pd

# Check intermediate values
if K is None or dist is None:
    raise Exception('Camera matrix or distortion coefficient not found')

MAX_FPS = 50
MIN_MATCHES = 100


def find_key_frames(video, idx1, idx2):
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
    keyframes = []
    traced_matches = None
    last_frame = None

    while success and curr_idx < idx2 - 1:
        success, frame = video.read()
        curr_idx += 1

        if curr_idx <= idx1:
            last_frame = frame
            continue

        # trace
        match_points_1, match_points_2, matches = get_points(last_frame, frame)  # TODO check if order is correct

        if traced_matches is None:
            traced_matches = [{
                'start_frame': curr_idx - 1,
                'coordinates': [match_points_1[i], match_points_2[i]],
                'from': x.queryIdx,
                'to': x.trainIdx} for i, x in enumerate(matches)]
        else:
            new_matches = dict([(x.queryIdx, x.trainIdx) for x in matches])
            for match in traced_matches:
                new_from = match['to']
                if new_from in new_matches:
                    match['from'] = new_from
                    match['to'] = new_matches[new_from]
                    match['coordinates'].append(match_points_2[list(new_matches.keys()).index(new_from)])
                else:
                    match['to'] = None

            traced_matches = list(filter(lambda m: m['to'] is not None, traced_matches))

            if len(traced_matches) <= MIN_MATCHES:
                # new keyframe
                print('found keyframe at pos ' + str(curr_idx))
                keyframes.append(traced_matches)
                traced_matches = None
        last_frame = frame

    keyframes.append(traced_matches)
    return keyframes


# get point correspondences
pts0 = pd.read_csv(REF_POINTS_0, sep=',', header=None, dtype=float).values
pts10 = pd.read_csv(REF_POINTS_10, sep=',', header=None, dtype=float).values
pts18 = pd.read_csv(REF_POINTS_18, sep=',', header=None, dtype=float).values
pts34 = pd.read_csv(REF_POINTS_34, sep=',', header=None, dtype=float).values
pts100 = pd.read_csv(REF_POINTS_100, sep=',', header=None, dtype=float).values

reader, writer = get_video_streams()
keyframes = find_key_frames(reader, 0, 35)

pts_list = []
for i in range(19):
    pts = []
    for frame in keyframes[0]:
        pts.append(frame['coordinates'][i])
    pts_list.append(pts)
pts_array = np.asarray(pts_list)

keyframe_pts = pts_array

R1, t1 = INIT_ORIENTATION, INIT_POSITION
R2, t2 = get_R_and_t(keyframe_pts[0], keyframe_pts[18], K)

# get world points of axis
axis, _ = get_3d_world_points(R1, t1, R2, t2, pts0, pts18, dist, K)
print(axis)

# get world points
_, world_coords = get_3d_world_points(R1, t1, R2, t2, keyframe_pts[0], keyframe_pts[18], dist, K)

ret, R, T, inliers = cv2.solvePnPRansac(world_coords, keyframe_pts[10], K, dist,
                                        reprojectionError=20.0)

points2d, _ = cv2.projectPoints(axis, R, T, K, dist)

img = cv2.imread(DATA_PATH + 'img_10.jpg')
draw_points(img, functools.reduce(operator.iconcat, points2d.astype(int).tolist(), []))
cv2.imshow('img', img)

cv2.waitKey(0)
