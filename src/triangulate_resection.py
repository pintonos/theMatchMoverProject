import functools
import operator

from functions import *
from util import *
import cv2
import pandas as pd

# Check intermediate values
if K is None or dist is None:
    raise Exception('Camera matrix or distortion coefficient not found')

MAX_FPS = 80
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


R0, t0 = INIT_ORIENTATION, INIT_POSITION

# get axis from frame 0 and 100
axis = get_3d_axis()

# first keyframe
R2, t2 = get_R_and_t(keyframe_pts[0][0], keyframe_pts[0][-1], K)
_, world_coords_1 = get_3d_world_points(R0, t0, R2, t2, keyframe_pts[0][0], keyframe_pts[0][-1], dist, K)
_, R2, t2, _ = cv2.solvePnPRansac(world_coords_1, keyframe_pts[0][0], K, dist, reprojectionError=20.0)
points2d_1, _ = cv2.projectPoints(axis, R2, t2, K, dist)

# second keyframe
R2 = cv2.Rodrigues(R2)[0]
R3, t3 = get_R_and_t(keyframe_pts[1][0], keyframe_pts[1][-1], K)
_, world_coords_2 = get_3d_world_points(R2, t2, R3, t3, keyframe_pts[1][0], keyframe_pts[1][-1], dist, K)
_, R3, t3, _ = cv2.solvePnPRansac(world_coords_2, keyframe_pts[1][0], K, dist, reprojectionError=20.0)
points2d_2, _ = cv2.projectPoints(axis, R3, t3, K, dist)

# third keyframe
R3 = cv2.Rodrigues(R3)[0]
R4, t4 = get_R_and_t(keyframe_pts[2][0], keyframe_pts[2][-1], K)
_, world_coords_3 = get_3d_world_points(R3, t3, R4, t4, keyframe_pts[2][0], keyframe_pts[2][-1], dist, K)
_, R4, t4, _ = cv2.solvePnPRansac(world_coords_3, keyframe_pts[2][0], K, dist, reprojectionError=20.0)
points2d_3, _ = cv2.projectPoints(axis, R4, t4, K, dist)

# draw
img1 = cv2.imread(DATA_PATH + 'img_0.jpg')
draw_points(img1, functools.reduce(operator.iconcat, points2d_1.astype(int).tolist(), []))
cv2.imshow('img_0', cv2.resize(img1, DEMO_RESIZE))

img2 = cv2.imread(DATA_PATH + 'img_18.jpg')
draw_points(img2, functools.reduce(operator.iconcat, points2d_2.astype(int).tolist(), []))
cv2.imshow('img_18', cv2.resize(img2, DEMO_RESIZE))

img2 = cv2.imread(DATA_PATH + 'img_34.jpg')
draw_points(img2, functools.reduce(operator.iconcat, points2d_3.astype(int).tolist(), []))
cv2.imshow('img_34', cv2.resize(img2, DEMO_RESIZE))
cv2.waitKey(0)


