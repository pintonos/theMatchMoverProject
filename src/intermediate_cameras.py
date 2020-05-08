import functools
import operator

from functions import *
import cv2
import pandas as pd

MAX_FPS = 20

keyframes, keyframe_idx = find_next_key_frame(0, MAX_FPS)
while keyframe_idx and keyframe_idx < MAX_FPS:
    if len(keyframes) > 1:
        range_idx = keyframe_idx - len(keyframes[-2][0]['coordinates'])
    else:
        range_idx = len(keyframes[-1][0]['coordinates'])

    tmp_kf, keyframe_idx = find_next_key_frame(keyframe_idx - (range_idx // 2), MAX_FPS)
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
R2, t2 = get_R_and_t(keyframe_pts[0][0], keyframe_pts[0][-1], K)
axis = get_3d_axis(R2, t2)

_, world_coords = get_3d_world_points(R0, t0, R2, t2, keyframe_pts[0][0], keyframe_pts[0][-1], dist, K)

reader, writer = get_video_streams()
frames_total = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
for i, frame in enumerate(keyframe_pts[0]):
    _, R, t, _ = cv2.solvePnPRansac(world_coords, keyframe_pts[0][i], K, dist, reprojectionError=1.0) # TODO check reprojectionError makes big difference!
    points2d, _ = cv2.projectPoints(axis, R, t, K, dist)

    _, img = reader.read()
    draw_points(img, functools.reduce(operator.iconcat, points2d.astype(int).tolist(), []))

    writer.write(img)

    cv2.imshow('img', cv2.resize(img, DEMO_RESIZE))
    cv2.waitKey(1000)

reader.release()
writer.release()
