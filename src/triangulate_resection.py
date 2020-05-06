import functools
import operator

from functions import *
from util import *
import cv2
import pandas as pd

# Check intermediate values
if K is None or dist is None:
    raise Exception('Camera matrix or distortion coefficient not found')

MAX_FPS = 40
MIN_MATCHES = 200


def find_next_key_frame(video, idx1, idx2):
    '''
    finds point matches that are preserved between idx1 and idx2
    :param video: video file with more than |idx2| frames
    :param idx1: index to start with
    :param idx2: index to end with
    :return: list of dict with structure [{'startPoint': 2DPoint, 'endPoint': 2DPoint}],
        startPoint is the 2DPoint of the first frame, endPoint is the 2DPoint of the last frame
    '''
    print("find keyframes between " + str(idx1) + " and " + str(idx2))
    if idx2 - idx1 <= 0:
        print("warning, called find_trace_points with 0 or negative frame indexes")
        return []

    curr_idx = -1
    success = True
    keyframes = []
    traced_matches = None
    prev_frame = None
    new_keyframe_pos = 0
    keyframe_found = False
    while success and curr_idx < idx2 - idx1 - 1:
        success, frame = video.read()
        curr_idx += 1

        if curr_idx <= idx1:
            prev_frame = frame
            continue

        # trace
        match_points_1, match_points_2, matches = get_points(prev_frame, frame)

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
                new_keyframe_pos = curr_idx + idx1 + 1
                print('found keyframe at pos ' + str(new_keyframe_pos))
                keyframes.append(traced_matches)
                keyframe_found = True
                break

        prev_frame = frame
    if keyframe_found:
        return keyframes, new_keyframe_pos
    else:
        print('no new keyframe found')
        return keyframes, None


# get point correspondences
pts0 = pd.read_csv(REF_POINTS_0, sep=',', header=None, dtype=float).values
pts10 = pd.read_csv(REF_POINTS_10, sep=',', header=None, dtype=float).values
pts18 = pd.read_csv(REF_POINTS_18, sep=',', header=None, dtype=float).values
pts34 = pd.read_csv(REF_POINTS_34, sep=',', header=None, dtype=float).values
pts100 = pd.read_csv(REF_POINTS_100, sep=',', header=None, dtype=float).values

reader, writer = get_video_streams()

keyframes, keyframe_idx = find_next_key_frame(reader, 0, MAX_FPS)
while keyframe_idx and keyframe_idx < MAX_FPS:
    if len(keyframes) > 1:
        range_idx = keyframe_idx - len(keyframes[-2][0]['coordinates'])
    else:
        range_idx = len(keyframes[-1][0]['coordinates'])

    tmp_kf, keyframe_idx = find_next_key_frame(reader, keyframe_idx - (range_idx // 2), MAX_FPS)
    keyframes = keyframes + tmp_kf

keyframe_pts = []
for k in keyframes:
    pts_list = []
    for i in range(len(k[0]['coordinates'])):
        pts = []
        for frame in k:
            pts.append(frame['coordinates'][i])
        pts_list.append(pts)
    keyframe_pts.append(np.asarray(pts_list))
'''
# check match points
img = cv2.imread(DATA_PATH + 'img_10.jpg')
draw_points(img, keyframe_pts[1][0])
cv2.imshow('img_10', cv2.resize(img, DEMO_RESIZE))

img = cv2.imread(DATA_PATH + 'img_38.jpg')
draw_points(img, keyframe_pts[1][-1])
cv2.imshow('img_38', cv2.resize(img, DEMO_RESIZE))

cv2.waitKey(0)'''

R0, t0 = INIT_ORIENTATION, INIT_POSITION
R2, t2 = get_R_and_t(keyframe_pts[0][0], keyframe_pts[0][-1], K)

# get world points of axis
axis, _ = get_3d_world_points(R0, t0, R2, t2, pts0, pts18, dist, K)
# print(axis)
# axis = get_3d_axis()
# print(axis)

# get world points
_, world_coords1 = get_3d_world_points(R0, t0, R2, t2, keyframe_pts[0][0], keyframe_pts[0][-1], dist, K)
_, R1, t1, _ = cv2.solvePnPRansac(world_coords1, keyframe_pts[0][9], K, dist,
                                  reprojectionError=20.0)

points2d, _ = cv2.projectPoints(axis, R1, t1, K, dist)

# works
#img = cv2.imread(DATA_PATH + 'img_9.jpg')
#draw_points(img, functools.reduce(operator.iconcat, points2d.astype(int).tolist(), []))
#cv2.imshow('img_9', cv2.resize(img, DEMO_RESIZE))


# check R2, t2 with new approach
R1, _ = cv2.Rodrigues(R1, dst=dist)
R3, t3 = get_R_and_t(keyframe_pts[1][0], keyframe_pts[1][-1], K)
_, world_coords2 = get_3d_world_points(R1, t1, R3, t3, keyframe_pts[1][0], keyframe_pts[1][-1], dist, K)

_, R2, t2, _ = cv2.solvePnPRansac(world_coords2, keyframe_pts[1][7], K, dist,
                                        reprojectionError=20.0)

points2d, _ = cv2.projectPoints(get_3d_axis(), R2, t2, K, dist) # TODO check 3d axis?

img = cv2.imread(DATA_PATH + 'img_18.jpg')
draw_points(img, functools.reduce(operator.iconcat, points2d.astype(int).tolist(), []))
cv2.imshow('img_18', cv2.resize(img, DEMO_RESIZE))
cv2.waitKey(0)


