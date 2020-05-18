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

start_frame = 100
end_frame = 150  # int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

keyframes, start_idx = get_all_keyframes(start_frame, end_frame)
keyframe_pts = get_keyframe_pts(keyframes)

print('keyframes at:', start_idx)

R0, t0 = INIT_ORIENTATION, INIT_POSITION
R2, t2 = get_R_and_t(keyframe_pts[0][0], keyframe_pts[0][-1], K)
axis = get_3d_axis(R2, t2, REF_POINTS_100, REF_POINTS_117)

'''
# P0
points2d_0, _ = cv2.projectPoints(axis, R0, t0, K, dist)
img0 = cv2.imread(DATA_PATH + 'img_0.jpg')
draw_points(img0, functools.reduce(operator.iconcat, points2d_0.astype(int).tolist(), []))
cv2.imshow('img_0', cv2.resize(img0, DEMO_RESIZE))

# P2
points2d_2, _ = cv2.projectPoints(axis, R2, t2, K, dist)
img2 = cv2.imread(DATA_PATH + 'img_17.jpg')
draw_points(img2, functools.reduce(operator.iconcat, points2d_2.astype(int).tolist(), []))
cv2.imshow('img_17', cv2.resize(img2, DEMO_RESIZE))
'''

# P1
_, world_coords1 = triangulate_points(R0, t0, R2, t2, keyframe_pts[0][0], keyframe_pts[0][-1], dist, K)
#_, R1, t1, _ = cv2.solvePnPRansac(world_coords1, keyframe_pts[0][-1], K, dist, reprojectionError=1)
_, R1, t1 = cv2.solvePnP(world_coords1, keyframe_pts[0][-1], K, dist)
print(R0)
print(cv2.Rodrigues(R1)[0])
print(t0)
print(t1)
points2d_1, _ = cv2.projectPoints(axis, R1, t1, K, dist)

img1 = cv2.imread(DATA_PATH + 'img_117.jpg')
draw_points(img1, functools.reduce(operator.iconcat, points2d_1.astype(int).tolist(), []))
cv2.imshow('img_117', cv2.resize(img1, DEMO_RESIZE))


# P3
'''
R1, _ = cv2.Rodrigues(R1)
_, world_coords3 = get_3d_world_points(R1, t1, R2, t2, keyframe_pts[1][0], keyframe_pts[1][-1], dist, K)
_, R3, t3, _ = cv2.solvePnPRansac(world_coords3, keyframe_pts[1][0], K, dist, reprojectionError=1.0)
points2d_3, _ = cv2.projectPoints(axis, R3, t3, K, dist)

img3 = cv2.imread(DATA_PATH + 'img_24.jpg')
draw_points(img3, functools.reduce(operator.iconcat, points2d_3.astype(int).tolist(), []))
cv2.imshow('img_24', cv2.resize(img3, DEMO_RESIZE))
'''

cv2.waitKey(0)







