from functions import *
import cv2
import os


def get_inlier_points(points_3d, points_2d, inliers):
    filtered_3d = []
    filtered_2d = []

    for i in range(len(inliers)):
        in_index = inliers[i][0]
        filtered_2d.append(points_2d[in_index])
        filtered_3d.append(points_3d[in_index])

    return np.asarray(filtered_3d), np.asarray(filtered_2d)


def get_intermediate_cameras(keyframe_cameras, points_3d, points_2d, frame_ranges):
    print('get intermediate cameras ...')
    all_cameras = []
    for i in range(len(keyframe_cameras)-1):
        all_cameras.append(keyframe_cameras[i])
        half_idx = start_idx[i + 1] - start_idx[i]
        for j in range(frame_ranges[i]):
            if j < half_idx - 1:
                _, R, t, _ = cv2.solvePnPRansac(points_3d[i], points_2d[i], K, dist, reprojectionError=1.0)
                all_cameras.append(Camera(R, t))
    return all_cameras


def correct_matches(points, start_idx):
    halfway_idx = start_idx[i] - start_idx[i - 1]
    if halfway_idx >= len(points):
        return None, None

    F, _ = get_F(points[0], points[-1])

    pts1 = np.reshape(points[0], (1, len(points[0]), 2))
    pts2 = np.reshape(points[halfway_idx], (1, len(points[halfway_idx]), 2))
    pts1, pts2 = cv2.correctMatches(F, pts1, pts2)

    return pts1[0], pts2[0]


reader, writer = get_video_streams()

start_frame = 0
end_frame = 160  # int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
keyframes_path = DATA_PATH + 'keyframes.npy'
start_idx_path = DATA_PATH + 'start_idx.npy'
keyframe_pts_path = DATA_PATH + 'keyframe_pts.npy'

# get keyframes
if os.path.isfile(keyframe_pts_path) and os.path.isfile(start_idx_path) and os.path.isfile(keyframes_path):
    keyframes = np.load(keyframes_path, allow_pickle=True)
    keyframe_pts = np.load(keyframe_pts_path, allow_pickle=True)
    start_idx = np.load(start_idx_path, allow_pickle=True)
else:
    keyframes, start_idx = get_all_keyframes(start_frame, end_frame)
    keyframe_pts = get_keyframe_pts(keyframes)
    # save data
    np.save(keyframes_path, keyframes)
    np.save(keyframe_pts_path, keyframe_pts)
    np.save(start_idx_path, start_idx)

print('found keyframes at positions:', start_idx)

# get cameras P0 and P2
R0, t0 = INIT_ORIENTATION, INIT_POSITION
R2, t2 = get_R_and_t(keyframe_pts[0][0], keyframe_pts[0][-1], K)
points_3d_0 = triangulate_points(R0, t0, R2, t2, keyframe_pts[0][0], keyframe_pts[0][-1], dist, K)

# compute P1 (halfway between P0 and P2)
halfway_idx = start_idx[1] - start_idx[0]
_, R, t, inliers = cv2.solvePnPRansac(points_3d_0, keyframe_pts[0][halfway_idx], K, dist, reprojectionError=1.0)
points_3d, points_2d = get_inlier_points(points_3d_0, keyframe_pts[0][halfway_idx], inliers)

# initialize lists
keyframe_cameras = [Camera(R0, t0), Camera(R, t)]
keyframe_world_points = [points_3d_0, points_3d]
keyframe_image_points = [keyframe_pts[0][0], points_2d]

# start resectioning
print('get keyframe cameras ...')
for i in range(1, len(keyframe_pts)):  # start iterating at camera P1

    pts1, pts2 = correct_matches(keyframe_pts[i], start_idx)
    if pts1 is None:
        break

    prev_cam = keyframe_cameras[i - 1]
    curr_cam = keyframe_cameras[i]

    # get next camera by resectioning form previous and current camera
    points_3d = triangulate_points(prev_cam.R_mat, prev_cam.t, curr_cam.R_mat, curr_cam.t, pts1, pts2, dist, K)
    _, R, t, inliers = cv2.solvePnPRansac(points_3d, keyframe_pts[i][-1], K, dist, reprojectionError=1.0)

    # filter points with inliers list
    points_3d, points_2d = get_inlier_points(points_3d, keyframe_pts[i][-1], inliers)

    # append to lists
    keyframe_cameras.append(Camera(R, t))
    keyframe_world_points.append(points_3d)
    keyframe_image_points.append(points_2d)

# bundle adjustment
# opt_cameras, opt_points_3d = start_bundle_adjustment(keyframe_cameras, keyframe_world_points, keyframe_image_points)
opt_cameras = keyframe_cameras
opt_points_3d = keyframe_world_points

# add intermediate cameras
frame_ranges = [len(keyframe_pts[i]) for i in range(len(keyframe_cameras)-1)]
cameras = get_intermediate_cameras(opt_cameras, opt_points_3d, keyframe_image_points, frame_ranges)

start, end = 0, 34
axis = get_3d_axis(cameras[start], start, cameras[end], end)
# save/show frames
for i in range(len(cameras)):
    _, img = reader.read()

    points_2d, _ = cv2.projectPoints(axis, cameras[i].R_mat, cameras[i].t, K, dist)
    draw_axis(img, points_2d)
    print('show frame:', i)
    cv2.imshow('normal', cv2.resize(img, DEMO_RESIZE))
    cv2.waitKey(0)
    writer.write(img)

reader.release()
writer.release()
