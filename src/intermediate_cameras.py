from functions import *
import cv2
import os


def project_inliers(R, t, world_coords, inliers):
    inlier_points3d = []
    for inlier_idx in inliers:
        inlier_points3d.append(world_coords[inlier_idx[0]])
    inlier_points3d = np.asarray(inlier_points3d)
    inlier_points_2d, _ = cv2.projectPoints(inlier_points3d, R, t, K, dist)

    return inlier_points3d, inlier_points_2d


def get_intermediate_cameras(keyframe_cameras, points_3d, keyframe_pts):
    cameras = []
    for i in range(len(keyframe_cameras)):
        cameras.append(keyframe_cameras[i])
        frame_range = len(keyframe_pts[i])
        for j in range(frame_range):
            _, R, t, _ = cv2.solvePnPRansac(points_3d[i], keyframe_pts[i][j], K, dist, reprojectionError=10)
            cameras.append(Camera(R, t))
    return cameras


reader, writer = get_video_streams()

start_frame = 0
end_frame = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
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

axis = get_3d_axis(R2, t2, REF_POINTS_0, REF_POINTS_18)
world_coords = get_3d_world_points(R0, t0, R2, t2, keyframe_pts[0][0], keyframe_pts[0][-1], dist, K)

keyframe_cameras = [Camera(R0, t0)]
keyframe_world_points = [world_coords]
keyframe_image_points = [keyframe_pts[0][0]]
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

        world_coords = get_3d_world_points(R_half, t_half, R, t, pts1[0], pts2[0], dist, K)
        keyframe_cameras.append(Camera(R, t))

    half_idx = start_idx[i+1] - start_idx[i]
    _, R_half, t_half, inliers_half = cv2.solvePnPRansac(world_coords, keyframe_pts[i][half_idx], K, dist, reprojectionError=5)
    _, R, t, inliers = cv2.solvePnPRansac(world_coords, keyframe_pts[i][-1], K, dist, reprojectionError=5)

    if inliers is None or inliers_half is None:
        print('no inliers found')
        break

    points_3d, points_2d = project_inliers(R, t, world_coords, inliers)
    keyframe_world_points.append(points_3d)
    keyframe_image_points.append(points_2d)


# bundle adjustment
# opt_cameras, opt_points_3d = start_bundle_adjustment(keyframe_cameras, keyframe_world_points, keyframe_image_points)

# add intermediate cameras
cameras = get_intermediate_cameras(keyframe_cameras, keyframe_world_points, keyframe_pts)

for i in range(len(cameras)):
    _, img = reader.read()
    img_opt = img.copy()

    points_2d, _ = cv2.projectPoints(axis, cameras[i].R, cameras[i].t, K, dist)
    draw_axis(img, points_2d)
    cv2.imshow('normal', cv2.resize(img, DEMO_RESIZE))
    cv2.waitKey(300)
    writer.write(img)

reader.release()
writer.release()
