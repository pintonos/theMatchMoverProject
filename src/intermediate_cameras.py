from functions import *
import cv2
import os

reader, writer = get_video_streams()

start_frame = 0
end_frame = 100  # int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
start_idx_path = DATA_PATH + 'start_idx.npy'
keyframe_pts_path = DATA_PATH + 'keyframe_pts.npy'

# get keyframes

if os.path.isfile(keyframe_pts_path) and os.path.isfile(start_idx_path):
    keyframe_pts = np.load(keyframe_pts_path, allow_pickle=True)
    start_idx = np.load(start_idx_path, allow_pickle=True)
else:
    keyframes, start_idx = get_all_keyframes(start_frame, end_frame)
    keyframe_pts = get_keyframe_pts(keyframes)
    # save data
    np.save(keyframe_pts_path, keyframe_pts)
    np.save(start_idx_path, start_idx)

print('keyframes at:', start_idx)

# get init cameras P0 and P2
R0, t0 = INIT_ORIENTATION, INIT_POSITION
R2, t2 = get_R_and_t(keyframe_pts[0][0], keyframe_pts[0][-1], K)

axis = get_3d_axis(R2, t2)
_, world_coords = get_3d_world_points(R0, t0, R2, t2, keyframe_pts[0][0], keyframe_pts[0][-1], dist, K)

for i, keyframe in enumerate(keyframe_pts):
    print('keyframe', i)
    # get new keyframe by resectioning
    if i != 0:
        R_half, _ = cv2.Rodrigues(R_half)
        R, _ = cv2.Rodrigues(R)

        F = get_F(keyframe_pts[i][0], keyframe_pts[i][-1])
        pts1 = np.reshape(keyframe_pts[i][0], (1, len(keyframe_pts[i][0]), 2))
        pts2 = np.reshape(keyframe_pts[i][half_idx], (1, len(keyframe_pts[i][half_idx]), 2))
        pts1, pts2 = cv2.correctMatches(F, pts1, pts2)

        _, world_coords = get_3d_world_points(R_half, t_half, R, t, pts1[0], pts2[0], dist, K)
        _, R, t, _ = cv2.solvePnPRansac(world_coords, keyframe_pts[i][-1], K, dist, reprojectionError=1)

    # fill between keyframes
    for j, frame in enumerate(keyframe_pts[i]):
        _, R, t, _ = cv2.solvePnPRansac(world_coords, keyframe_pts[i][j], K, dist, reprojectionError=1)

        half_idx = start_idx[i+1]-start_idx[i]
        if j == half_idx:
            R_half, t_half = R, t

        if j < half_idx or i == len(keyframe_pts)-1:
            points2d, _ = cv2.projectPoints(axis, R, t, K, dist)

            _, img = reader.read()
            draw_axis(img, points2d)

            writer.write(img)

            cv2.imshow('img', cv2.resize(img, DEMO_RESIZE))
            cv2.waitKey(200)


reader.release()
writer.release()
