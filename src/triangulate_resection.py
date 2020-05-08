import functools
import operator

from functions import *
from util import *
import cv2

# Check intermediate values
if K is None or dist is None:
    raise Exception('Camera matrix or distortion coefficient not found')


MIN_MATCHES = 200


def find_next_key_frame(idx1, idx2):
    """
    finds point matches that are preserved between idx1 and idx2
    :param video: video file with more than |idx2| frames
    :param idx1: index to start with
    :param idx2: index to end with
    :return: list of dict with structure [{'startPoint': 2DPoint, 'endPoint': 2DPoint}],
        startPoint is the 2DPoint of the first frame, endPoint is the 2DPoint of the last frame
    """

    video, _ = get_video_streams()
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
    video.release()
    if keyframe_found:
        return keyframes, new_keyframe_pos
    else:
        print('no new keyframe found')
        return keyframes, None


reader, writer = get_video_streams()
MAX_FPS = 80  # int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

keyframes = np.load(DATA_PATH + 'keys.npy', allow_pickle=True)
"""
keyframes, keyframe_idx = find_next_key_frame(0, MAX_FPS)
while keyframe_idx and keyframe_idx < MAX_FPS:
    if len(keyframes) > 1:
        range_idx = keyframe_idx - len(keyframes[-2][0]['coordinates'])
    else:
        range_idx = len(keyframes[-1][0]['coordinates'])

    tmp_kf, keyframe_idx = find_next_key_frame(keyframe_idx - (range_idx // 2), MAX_FPS)
    keyframes = keyframes + tmp_kf

# np.save(DATA_PATH + 'keys.npy', keyframes)
"""

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
R2, t2 = get_R_and_t(keyframe_pts[0][0], keyframe_pts[0][-1], K)
axis = get_3d_axis(R2, t2)

# P0

points2d_1, _ = cv2.projectPoints(axis, R0, t0, K, dist)

img0 = cv2.imread(DATA_PATH + 'img_0.jpg')
draw_points(img0, functools.reduce(operator.iconcat, points2d_1.astype(int).tolist(), []))
cv2.imshow('img_0', cv2.resize(img0, DEMO_RESIZE))

# P2
points2d_2, _ = cv2.projectPoints(axis, R2, t2, K, dist)

img2 = cv2.imread(DATA_PATH + 'img_18.jpg')
draw_points(img2, functools.reduce(operator.iconcat, points2d_2.astype(int).tolist(), []))
cv2.imshow('img_18', cv2.resize(img2, DEMO_RESIZE))

# P1
_, world_coords1 = get_3d_world_points(R0, t0, R2, t2, keyframe_pts[0][0], keyframe_pts[0][-1], dist, K)
_, R1, t1, _ = cv2.solvePnPRansac(world_coords1, keyframe_pts[0][9], K, dist, reprojectionError=20.0)
points2d_1, _ = cv2.projectPoints(axis, R1, t1, K, dist)

img1 = cv2.imread(DATA_PATH + 'img_9.jpg')
draw_points(img1, functools.reduce(operator.iconcat, points2d_1.astype(int).tolist(), []))
cv2.imshow('img_9', cv2.resize(img1, DEMO_RESIZE))

# P3
R1, _ = cv2.Rodrigues(R1)
_, world_coords3 = get_3d_world_points(R1, t1, R2, t2, keyframe_pts[1][0], keyframe_pts[1][-1], dist, K)
_, R3, t3, _ = cv2.solvePnPRansac(world_coords3, keyframe_pts[1][0], K, dist, reprojectionError=20.0)
points2d_1, _ = cv2.projectPoints(axis, R3, t3, K, dist)

img3 = cv2.imread(DATA_PATH + 'img_34.jpg')
draw_points(img3, functools.reduce(operator.iconcat, points2d_1.astype(int).tolist(), []))
cv2.imshow('img_34', cv2.resize(img3, DEMO_RESIZE))
cv2.waitKey(0)
