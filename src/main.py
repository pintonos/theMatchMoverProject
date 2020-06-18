from functions import *
from util import helper
import numpy as np
import cv2
import logging


# video streams for input and output
reader, writer = helper.get_video_streams()

# restrict video to frame numbers
start_frame = 0
end_frame = 100  # int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

keyframes, keyframe_idx = load_keyframes(start_frame, end_frame)
keyframe_pts = get_keyframe_pts(keyframes)

logging.info('found keyframes at positions: {0}'.format(keyframe_idx))

# get cameras P0 and P2
R0, t0 = np.identity(3), np.asarray([0, 0, 0], dtype=float)  # init position and orientation
R2, t2 = get_R_and_t(keyframe_pts[0][0], keyframe_pts[0][-1], K)
points_3d_0 = triangulate_points(R0, t0, R2, t2, keyframe_pts[0][0], keyframe_pts[0][-1], dist, K)

# compute P1 (halfway between P0 and P2)
halfway_idx = keyframe_idx[1] - keyframe_idx[0]
_, R, t, inliers = cv2.solvePnPRansac(points_3d_0, keyframe_pts[0][halfway_idx], K, dist, reprojectionError=2.0)
points_3d, points_2d = get_inlier_points(points_3d_0, keyframe_pts[0], inliers)

P0 = Camera(R0, t0)
P1 = Camera(R, t)
points_3d = get_3d_points_for_consecutive_frames(points_3d, points_2d)

# initialize lists
keyframe_cameras = [P0, P1]
keyframe_world_points = [points_3d]
keyframe_image_points = [points_2d]

# start resectioning
logging.info('get keyframe cameras ...')
for i in range(2, len(keyframe_pts) + 1):  # start iterating at camera P1

    halfway_idx = keyframe_idx[i - 1] - keyframe_idx[i - 2]
    pts1, pts2 = correct_matches(keyframe_pts[i-2], halfway_idx)
    if pts1 is None:
        break

    prev_cam = keyframe_cameras[-2]
    curr_cam = keyframe_cameras[-1]

    # get next camera by resectioning from previous and current camera
    points_3d = triangulate_points(prev_cam.R_mat, prev_cam.t, curr_cam.R_mat, curr_cam.t, pts1, pts2, dist, K)
    _, R, t, inliers = cv2.solvePnPRansac(points_3d, keyframe_pts[i-2][-1], K, dist, reprojectionError=2.0)
    logging.info('{0} inliers in keyframe {1}'.format(len(inliers), i))

    # filter points with inliers list
    points_3d, points_2d = get_inlier_points(points_3d, keyframe_pts[i-2], inliers)
    points_3d = get_3d_points_for_consecutive_frames(points_3d, points_2d)

    # append to lists
    keyframe_cameras.append(Camera(R, t))
    keyframe_world_points.append(points_3d)
    keyframe_image_points.append(points_2d)


# add intermediate cameras
cameras = get_intermediate_cameras(keyframe_cameras, keyframe_world_points, keyframe_image_points, keyframe_idx)

# read points of artificial object
start, end = 0, 76
axis = get_3d_points_from_ref(cameras[start], start, cameras[end], end)

# save/show frames
for i in range(len(cameras)):
    _, img = reader.read()
    points_2d = get_cube_points_from_axis_points(cameras[i], axis)
    draw_cube(img, points_2d)
    logging.info('process frame {0}'.format(i))
    if SHOW_FRAMES == 'True':
        cv2.imshow('normal', cv2.resize(img, (960, 540)))
        cv2.waitKey(0)
    writer.write(img)

logging.info('done')
cv2.destroyAllWindows()
reader.release()
writer.release()
sys.exit(0)
