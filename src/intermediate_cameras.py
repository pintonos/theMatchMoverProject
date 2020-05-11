import functools
import operator

from functions import *
import cv2

reader, writer = get_video_streams()
MAX_FPS = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

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

for i, keyframe in enumerate(keyframe_pts):
    print("i", i)
    # new keyframe by resectioning
    if i != 0:
        R_half, _ = cv2.Rodrigues(R_half)
        R, _ = cv2.Rodrigues(R)

        F = get_F(keyframe_pts[i][0], keyframe_pts[i][-1])
        pts1 = np.reshape(keyframe_pts[i][0], (1, len(keyframe_pts[i][0]), 2))
        pts2 = np.reshape(keyframe_pts[i][len(keyframe_pts[i])//2 - 1], (1, len(keyframe_pts[i][len(keyframe_pts[i])//2 - 1]), 2))
        pts1, pts2 = cv2.correctMatches(F, pts1, pts2)

        _, world_coords = get_3d_world_points(R_half, t_half, R, t, pts1[0], pts2[0], dist, K)
        _, R, t, _ = cv2.solvePnPRansac(world_coords, keyframe_pts[i][-1], K, dist, reprojectionError=1.0)

    # fill between keyframes
    for j, frame in enumerate(keyframe_pts[i]):
        _, R, t, _ = cv2.solvePnPRansac(world_coords, keyframe_pts[i][j], K, dist, reprojectionError=1.0) # TODO check reprojectionError makes big difference!

        if j == len(keyframe_pts[i]) // 2:
            R_half, t_half = R, t

        if j < len(keyframe_pts[i]) // 2 or i == len(keyframe_pts)-1:
            print("j", j)
            points2d, _ = cv2.projectPoints(axis, R, t, K, dist)

            _, img = reader.read()
            draw_axis(img, points2d)

            writer.write(img)

            cv2.imshow('img', cv2.resize(img, DEMO_RESIZE))
            cv2.waitKey(500)


reader.release()
writer.release()
