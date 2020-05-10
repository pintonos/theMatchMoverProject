import functools
import operator
from itertools import repeat

from functions import *
from util import *
import cv2

# Check intermediate values
if K is None or dist is None:
    raise Exception('Camera matrix or distortion coefficient not found')

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

points2d_0, _ = cv2.projectPoints(axis, R0, t0, K, dist)
'''
img0 = cv2.imread(DATA_PATH + 'img_0.jpg')
draw_points(img0, functools.reduce(operator.iconcat, points2d_0.astype(int).tolist(), []))
cv2.imshow('img_0', cv2.resize(img0, DEMO_RESIZE))
'''
# P2
points2d_2, _ = cv2.projectPoints(axis, R2, t2, K, dist)
'''
img2 = cv2.imread(DATA_PATH + 'img_18.jpg')
draw_points(img2, functools.reduce(operator.iconcat, points2d_2.astype(int).tolist(), []))
cv2.imshow('img_18', cv2.resize(img2, DEMO_RESIZE))
'''
# P1
_, world_coords1 = get_3d_world_points(R0, t0, R2, t2, keyframe_pts[0][0], keyframe_pts[0][-1], dist, K)
_, R1, t1, _ = cv2.solvePnPRansac(world_coords1, keyframe_pts[0][9], K, dist, reprojectionError=20.0)
points2d_1, _ = cv2.projectPoints(axis, R1, t1, K, dist)
'''
img1 = cv2.imread(DATA_PATH + 'img_9.jpg')
draw_points(img1, functools.reduce(operator.iconcat, points2d_1.astype(int).tolist(), []))
cv2.imshow('img_9', cv2.resize(img1, DEMO_RESIZE))
'''
# P3
R1, _ = cv2.Rodrigues(R1)
_, world_coords3 = get_3d_world_points(R1, t1, R2, t2, keyframe_pts[1][0], keyframe_pts[1][-1], dist, K)
_, R3, t3, _ = cv2.solvePnPRansac(world_coords3, keyframe_pts[1][0], K, dist, reprojectionError=20.0)
points2d_3, _ = cv2.projectPoints(axis, R3, t3, K, dist)
'''
img3 = cv2.imread(DATA_PATH + 'img_34.jpg')
draw_points(img3, functools.reduce(operator.iconcat, points2d_3.astype(int).tolist(), []))
cv2.imshow('img_34', cv2.resize(img3, DEMO_RESIZE))
cv2.waitKey(0)'''


cameras = [Camera(R0, t0), Camera(R2, t2)]
adjusted_3d_coords = startBundleAdjustment(cameras, axis, np.asarray([points2d_0, points2d_2]))
print(axis)
print(adjusted_3d_coords)






