from util import *


def get_intermediate_cameras(keyframe_cameras, points_3d, points_2d, start_idx):
    print('get intermediate cameras ...')
    all_cameras = []

    j = -1  # keyframe index
    k = 0  # frame index
    for i in range(start_idx[0], start_idx[-1]):
        if i in start_idx[:-1]:  # keyframe
            j += 1
            if j > 0:
                k = start_idx[j] - start_idx[j - 1]  # other keyframes
            all_cameras.append(keyframe_cameras[j])
        else:  # intermediate frame
            _, R, t, inliers = cv2.solvePnPRansac(points_3d[j][k], points_2d[j][k], K, dist, reprojectionError=1.5)
            all_cameras.append(Camera(R, t))
            k += 1

    # last frame camera
    all_cameras.append(keyframe_cameras[-1])

    return all_cameras
