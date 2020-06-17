from functions import *
from util import *
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import time

"""
bundle adjustments, from:
https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
"""


def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj ** 2, axis=1)
    r = 1 + k1 * n + k2 * n ** 2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def get_residuals(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A


def build_camera(R, t):
    """
    First 3 components in each row form a rotation vector,
    next 3 components form a translation vector,
    then a focal distance and two distortion parameters.
    """
    camera = R.flatten()
    camera = np.append(camera, t.flatten())
    camera = np.append(camera, K[0][0])
    camera = np.append(camera, dist[0, 0:2].flatten())
    return camera


def revert_camera_build(bundle_camera_matrix):
    R = np.asarray(bundle_camera_matrix[0:3]).T.reshape(-1, 1)
    t = np.asarray(bundle_camera_matrix[3:6]).T.reshape(-1, 1)
    return R, t


def get_close_matchpoints(match_points1, match_points2, threshold_x=5, threshold_y=5):
    close_idx1 = []
    close_idx2 = []
    for i, pt in enumerate(match_points1):
        for j, pt2 in enumerate(match_points2):
            if pt[0] + threshold_x > pt2[0] > pt[0] - threshold_x and pt[1] + threshold_y > pt2[1] > pt[1] - threshold_y:
                close_idx1.append(i)
                close_idx2.append(j)
    return close_idx1, close_idx2


def prepare_data(cameras, frame_points_3d, frame_points_2d, keyframe_idx):
    camera_params = np.empty((0, 9))
    for c in cameras:
        R, _ = cv2.Rodrigues(c.R_mat)
        camera = build_camera(R, c.t)
        camera_params = np.append(camera_params, [camera], axis=0)

    camera_indices = []
    point_indices = []
    points_2d = np.empty((0, 2))
    points_3d = np.empty((0, 3))

    camera_id = 0
    pt_id_counter = 0
    for k, pts_2d in enumerate(frame_points_2d):
        if k > 0:
            halfway_idx = keyframe_idx[k] - keyframe_idx[k - 1] - 1
            points_2d = np.vstack((points_2d, frame_points_2d[k-1][halfway_idx]))
            points_3d = np.vstack((points_3d, frame_points_3d[k-1][halfway_idx]))
            camera_indices += [camera_id for _ in range(len(frame_points_2d[k-1][halfway_idx]))]
            point_indices += [i for i in range(pt_id_counter, pt_id_counter + len(frame_points_2d[k-1][halfway_idx]))]
            pt_id_counter = pt_id_counter + len(frame_points_2d[k-1][halfway_idx])

        if k > 1:
            end_idx = keyframe_idx[k + 1] - keyframe_idx[k - 1] - 3
            points_2d = np.vstack((points_2d, frame_points_2d[k-2][end_idx]))
            points_3d = np.vstack((points_3d, frame_points_3d[k-2][end_idx]))
            camera_indices += [camera_id for _ in range(len(frame_points_2d[k-2][end_idx]))]
            point_indices += [i for i in range(pt_id_counter, pt_id_counter + len(frame_points_2d[k-2][end_idx]))]
            pt_id_counter = pt_id_counter + len(frame_points_2d[k-2][end_idx])

        points_2d = np.vstack((points_2d, frame_points_2d[k][0]))
        points_3d = np.vstack((points_3d, frame_points_3d[k][0]))
        camera_indices += [camera_id for _ in range(pts_2d.shape[1])]
        point_indices += [i for i in range(pt_id_counter, pt_id_counter + pts_2d.shape[1])]

        camera_id += 1
        pt_id_counter = pt_id_counter + pts_2d.shape[1]

    return camera_params, np.asarray(camera_indices), np.asarray(point_indices), points_3d, points_2d


def optimized_params(params, n_cameras, n_points_per_frame):
    """
    Retrieve camera parameters and 3-D coordinates.
    """
    tmp = params[:n_cameras * 9].reshape((n_cameras, 9))
    cameras = []
    for c in tmp:
        R, t = revert_camera_build(c)
        cameras.append(Camera(R, t))

    points3d = []
    range_counter = 0
    for range in n_points_per_frame:
        range_points = range * 3
        points3d.append(
            params[n_cameras * 9 + range_counter:n_cameras * 9 + range_counter + range_points].reshape((range, 1, 3)))
        range_counter += range_points

    return cameras, points3d


def filter_outliers(cameras, points_2d, points_3d, keyframe_idx):
    in_points_2d = []
    in_points_3d = []

    all_indices = []
    indices = [i for i in range(1000)]
    for i in range(0, len(cameras)):
        reprojected, _ = cv2.projectPoints(np.asarray(points_3d[i][0]), cameras[i].R_vec, cameras[i].t, K, dist)
        reprojected = np.reshape(reprojected, (len(reprojected), 2))

        close_arr = np.isclose(points_2d[i][0], reprojected, atol=50)

        current_indices = []
        for j in range(0, len(close_arr)):
            if close_arr[j][0] and close_arr[j][1]:
                current_indices.append(j)

        tmp_2d = []
        tmp_3d = []
        for k, _ in enumerate(points_2d[i]):
            tmp_2d.append(points_2d[i][k][current_indices])
            tmp_3d.append(points_3d[i][k][current_indices])

        in_points_2d.append(np.asarray(tmp_2d))
        in_points_3d.append(np.asarray(tmp_3d))

    return in_points_3d, in_points_2d


def start_bundle_adjustment(cameras, points3d, points2d, keyframe_idx):
    print('start bundle adjustment ...')
    #points3d, points2d = filter_outliers(cameras, points2d, points3d, keyframe_idx)
    camera_params, camera_indices, point_indices, points_3d, points_2d = prepare_data(cameras, points3d, points2d,
                                                                                      keyframe_idx)

    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]
    n_points_per_frame = [point.shape[0] for point in points2d]

    n = 9 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]
    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

    f0 = get_residuals(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
    plt.plot(f0)
    plt.show()

    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    t0 = time.time()
    res = least_squares(get_residuals, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-5, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    t1 = time.time()

    print("Optimization took {0:.0f} seconds".format(t1 - t0))

    plt.plot(res.fun)
    plt.show()

    optimized_cameras, optimized_points_3d = optimized_params(res.x, n_cameras, n_points_per_frame)

    return optimized_cameras, optimized_points_3d

