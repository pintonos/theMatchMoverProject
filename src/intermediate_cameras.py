from functions import *
import cv2
import os
from functools import reduce


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


def filter_tracing(tracing, inliers, id):
    inliers = np.squeeze(inliers, axis=1)
    tracing[id] = [tracing[id][i] for i in inliers]
    return tracing


def get_inlier_points_simple(points_3d, points_2d, inliers):
    filtered_3d = []
    filtered_2d = []

    for i in range(len(inliers)):
        filtered_2d.append(points_2d[i])
        filtered_3d.append(points_3d[i])

    return np.asarray(filtered_3d), np.asarray(filtered_2d)


def get_intermediate_cameras(keyframe_cameras, points_3d, points_2d, start_idx):
    print('get intermediate cameras ...')
    all_cameras = []
    all_3d_points = []
    all_2d_points = []
    for i in range(len(keyframe_cameras)-2):
        half_idx = start_idx[i + 1] - start_idx[i]
        inliers_list = []
        camera_list = []
        if i != 0:
            start = half_idx
            end = len(points_2d[i])
        else:
            start = 0
            end = half_idx - 1
        for j in range(start, end):
            _, R, t, inliers = cv2.solvePnPRansac(points_3d[i][j], points_2d[i][j], K, dist, reprojectionError=3.0)
            camera_list.append(Camera(R, t))
            inliers_list.append(np.asarray(inliers).flatten())
        filtered_inliers = reduce(np.intersect1d, (inliers_list))

        # keyframe camera
        pts1, pts2 = get_inlier_points_simple(points_3d[i][0], points_2d[i][0], filtered_inliers)
        all_cameras.append(keyframe_cameras[i])
        all_3d_points.append(pts1)
        all_2d_points.append(pts2)

        # intermediate cameras
        for k, j in enumerate(range(start, end)):
            pts1, pts2 = get_inlier_points_simple(points_3d[i][j], points_2d[i][j], filtered_inliers)
            all_cameras.append(camera_list[k])
            #all_cameras.append(keyframe_cameras[i]) # TODO test why keyframes only better?
            all_3d_points.append(pts1)
            all_2d_points.append(pts2)
    return all_cameras, all_3d_points, all_2d_points


def correct_matches(points, start_idx, i):
    halfway_idx = start_idx[i] - start_idx[i - 1]
    if halfway_idx >= len(points):
        return None, None

    F, _ = get_F(points[0], points[-1])

    pts1 = np.reshape(points[0], (1, len(points[0]), 2))
    pts2 = np.reshape(points[halfway_idx], (1, len(points[halfway_idx]), 2))
    pts1, pts2 = cv2.correctMatches(F, pts1, pts2)

    return pts1[0], pts2[0]


def get_3d_points_for_consecutive_frames(points_3d, prev_cam, curr_cam, points_2d): # TODO refactor, needed?
    # fill missing 3d points for consecutive frames
    points_3d = [points_3d for i in range(points_2d.shape[0])]
    '''original_points_3d = points_3d
    points_3d = [points_3d]
    start_2d_points = np.swapaxes(points_2d[0], 0, 1)
    for frame in range(1, points_2d.shape[0]):
        if frame < points_2d.shape[0]//2:
            points_3d.append(original_points_3d)
        else:
            frame_2d_points = np.swapaxes(points_2d[frame], 0, 1)
            hom = cv2.triangulatePoints(get_P(prev_cam.R_mat, prev_cam.t, K), get_P(curr_cam.R_mat, curr_cam.t, K),
                                        start_2d_points, frame_2d_points)
            hom = (hom / hom[3])[:-1].transpose()
            points_3d.append(np.asarray(hom))'''
    return np.asarray(points_3d)


reader, writer = get_video_streams()

keyframes_only = False

start_frame = 0
end_frame = 70  # int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
keyframes_path = DATA_PATH + 'keyframes.npy'
start_idx_path = DATA_PATH + 'start_idx.npy'
keyframe_pts_path = DATA_PATH + 'keyframe_pts.npy'


# trace points and find keyframes
if os.path.isfile(keyframe_pts_path) and os.path.isfile(start_idx_path) and os.path.isfile(keyframes_path):
    tracing = np.load(keyframes_path, allow_pickle=True)
    keyframe_pts = np.load(keyframe_pts_path, allow_pickle=True)
    keyframe_idx = np.load(start_idx_path, allow_pickle=True)
else:
    tracing = trace_points(start_frame, end_frame)
    keyframe_idx = find_keyframes(tracing)
    keyframe_pts = get_keyframe_pts(tracing, keyframe_idx)
    # save data
    np.save(keyframes_path, tracing)
    np.save(keyframe_pts_path, keyframe_pts)
    np.save(start_idx_path, keyframe_idx)
keyframe_idx = find_keyframes(tracing)
keyframe_pts = get_keyframe_pts(tracing, keyframe_idx)
print('found keyframes at positions:', keyframe_idx)

keyframe_pts = get_keyframe_pts(tracing, keyframe_idx)

# get cameras P0 and P2
R0, t0 = INIT_ORIENTATION, INIT_POSITION
R2, t2 = get_R_and_t(keyframe_pts[0][0], keyframe_pts[0][-1], K)
points_3d_0 = triangulate_points(R0, t0, R2, t2, keyframe_pts[0][0], keyframe_pts[0][-1], dist, K)

# compute P1 (halfway between P0 and P2)
halfway_idx = keyframe_idx[1] - keyframe_idx[0]
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

    pts1, pts2 = correct_matches(keyframe_pts[i-1], keyframe_idx, i)
    if pts1 is None:
        break

    prev_cam = keyframe_cameras[i - 1]
    curr_cam = keyframe_cameras[i]

    # get next camera by resectioning form previous and current camera
    points_3d = triangulate_points(prev_cam.R_mat, prev_cam.t, curr_cam.R_mat, curr_cam.t, pts1, pts2, dist, K)
    _, R, t, inliers = cv2.solvePnPRansac(points_3d, keyframe_pts[i-1][-1], K, dist, reprojectionError=2.0)
    print(i, '\t', len(inliers))

    # filter points with inliers list
    points_3d, points_2d = get_inlier_points(points_3d, keyframe_pts[i-1], inliers)
    filtered_tracing = filter_tracing(tracing, inliers, keyframe_idx[i-1])
    points_3d = get_3d_points_for_consecutive_frames(points_3d, prev_cam, curr_cam, points_2d)

    # append to lists
    keyframe_cameras.append(Camera(R, t))
    keyframe_world_points.append(points_3d)
    keyframe_image_points.append(points_2d)

if keyframes_only:
    # opt_cameras, opt_points_3d = start_bundle_adjustment(keyframe_cameras, keyframe_world_points, keyframe_image_points, keyframe_idx)
    # cameras = opt_cameras
    cameras = keyframe_cameras

    # only show keyframes
    axis = get_3d_points_from_ref(cameras[0], 0, cameras[3], 30)
    keyframe_idx = np.append(keyframe_idx, end_frame)
    for i, c in enumerate(cameras):
        img = get_frame(keyframe_idx[i])
        points_2d = get_cube_points_from_axis_points(cameras[i], axis)
        draw_cube(img, points_2d)
        #draw_points(img, functools.reduce(operator.iconcat, keyframe_image_points[i].astype(int).tolist(), []))
        print('show frame:', keyframe_idx[i])
        cv2.imshow('normal', cv2.resize(img, DEMO_RESIZE))
        cv2.waitKey(0)

else:
    # add intermediate cameras
    cameras, points_3d, points_2d = get_intermediate_cameras(keyframe_cameras, keyframe_world_points, keyframe_image_points, keyframe_idx)

    # bundle adjustment for each keyframe
    '''
    opt_cameras, opt_points_3d = [], []
    for i, idx in enumerate(start_idx[1:]):
        last_idx = start_idx[i]
        opt_cameras_tmp, opt_points_3d_tmp = start_bundle_adjustment(cameras[last_idx:idx], points_3d[last_idx:idx], points_2d[last_idx:idx], [0, idx-last_idx])
        opt_cameras = opt_cameras + opt_cameras_tmp
        opt_points_3d = opt_points_3d + opt_points_3d_tmp
    '''

    start, end = 0, 30
    axis = get_3d_points_from_ref(cameras[start], start, cameras[32], end)

    # save/show frames
    for i in range(len(cameras)):
        _, img = reader.read()
        points_2d = get_cube_points_from_axis_points(cameras[i], axis)
        draw_cube(img, points_2d)
        #points_2d, _ = cv2.projectPoints(axis, cameras[i].R_mat, cameras[i].t, K, dist)
        #draw_axis(img, points_2d)
        print('show frame:', i)
        cv2.imshow('normal', cv2.resize(img, DEMO_RESIZE))
        cv2.waitKey(0)
        writer.write(img)



reader.release()
writer.release()
