from functions import *
import cv2
import os


def get_inlier_points(points_3d, points_2d, inliers):
    filtered_3d = []
    filtered_2d = []

    points_2d = np.swapaxes(points_2d, 0, 1)
    for i in range(len(inliers)):
        in_index = inliers[i][0]
        filtered_2d.append(points_2d[in_index])
        filtered_3d.append(points_3d[in_index])

    filtered_2d = np.swapaxes(np.asarray(filtered_2d), 0, 1)
    return np.asarray(filtered_3d), filtered_2d


def get_inlier_points_simple(points_3d, points_2d, inliers):
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
    all_3d_points = []
    all_2d_points = []
    for i in range(len(keyframe_cameras)-1):
        all_cameras.append(keyframe_cameras[i])
        all_3d_points.append(points_3d[i][0])
        all_2d_points.append(points_2d[i][0])
        half_idx = start_idx[i + 1] - start_idx[i]
        for j in range(1, frame_ranges[i]):
            if j < half_idx:
                _, R, t, inliers = cv2.solvePnPRansac(points_3d[i][j], points_2d[i][j], K, dist, reprojectionError=2.0)
                frame_in_3d, frame_in_2d = get_inlier_points_simple(points_3d[i][j], points_2d[i][j], inliers)
                all_cameras.append(Camera(R, t))
                all_3d_points.append(frame_in_3d)
                all_2d_points.append(frame_in_2d)
    return all_cameras, all_3d_points, all_2d_points


def correct_matches(points, start_idx):
    halfway_idx = start_idx[i] - start_idx[i - 1]
    if halfway_idx >= len(points):
        return None, None

    F, _ = get_F(points[0], points[-1])

    pts1 = np.reshape(points[0], (1, len(points[0]), 2))
    pts2 = np.reshape(points[halfway_idx], (1, len(points[halfway_idx]), 2))
    pts1, pts2 = cv2.correctMatches(F, pts1, pts2)

    return pts1[0], pts2[0]


def get_3d_points_for_consecutive_frames(points_3d, prev_cam, curr_cam, points_2d):
    # fill missing 3d points for consecutive frames
    points_3d = [points_3d]
    start_2d_points = np.swapaxes(points_2d[0], 0, 1)
    for frame in range(1, points_2d.shape[0]):
        frame_2d_points = np.swapaxes(points_2d[frame], 0, 1)
        hom = cv2.triangulatePoints(get_P(prev_cam.R_mat, prev_cam.t, K), get_P(curr_cam.R_mat, curr_cam.t, K),
                                    start_2d_points, frame_2d_points)
        hom = (hom / hom[3])[:-1].transpose()
        points_3d.append(np.asarray(hom))
    return np.asarray(points_3d)


reader, writer = get_video_streams()

start_frame = 0
end_frame = 50  # int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
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
points_3d, points_2d = get_inlier_points(points_3d_0, keyframe_pts[0][:halfway_idx], inliers)

P0 = Camera(R0, t0)
P1 = Camera(R, t)
points_3d = get_3d_points_for_consecutive_frames(points_3d, P0, P1, points_2d)

# initialize lists
keyframe_cameras = [P0, P1]
keyframe_world_points = [points_3d]
keyframe_image_points = [points_2d]

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
    _, R, t, inliers = cv2.solvePnPRansac(points_3d, keyframe_pts[i][-1], K, dist, reprojectionError=5.0)

    # filter points with inliers list
    points_3d, points_2d = get_inlier_points(points_3d, keyframe_pts[i], inliers)
    points_3d = get_3d_points_for_consecutive_frames(points_3d, prev_cam, curr_cam, points_2d)

    # append to lists
    keyframe_cameras.append(Camera(R, t))
    keyframe_world_points.append(points_3d)
    keyframe_image_points.append(points_2d)

# add intermediate cameras
frame_ranges = [len(keyframe_pts[i]) for i in range(len(keyframe_cameras)-1)]
cameras, points_3d, points_2d = get_intermediate_cameras(keyframe_cameras, keyframe_world_points, keyframe_image_points, frame_ranges)

# bundle adjustment
opt_cameras, opt_points_3d = start_bundle_adjustment(cameras, points_3d, points_2d, start_idx)

start, end = 0, 34
axis = get_3d_axis(cameras[start], start, cameras[end], end)

# save/show frames
cameras = opt_cameras
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
