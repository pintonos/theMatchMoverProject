from itertools import repeat
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from functions import *
from util import *

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
    camera = np.append(camera, dist[0, 0:2].flatten())  # TODO our dist in 2 parameter?
    return camera


def revert_camera_build(bundle_camera_matrix):
    R = np.asarray(bundle_camera_matrix[0:3]).T.reshape(-1,1)
    t = np.asarray(bundle_camera_matrix[3:6]).T.reshape(-1,1)
    return R, t


def prepare_data(cameras, frame_points_3d, frame_points_2d):
    camera_params = np.empty((0, 9))
    for c in cameras:
        R, _ = cv2.Rodrigues(c.R_mat)
        camera = build_camera(R, c.t)
        camera_params = np.append(camera_params, [camera], axis=0)

    camera_indices = np.empty(0, dtype=int)
    point_indices = np.empty(0, dtype=int)

    points_2d = np.empty((0, 2))
    for i, pts_2d in enumerate(frame_points_2d):
        camera_indices = np.append(camera_indices, np.asarray(list(repeat(i, len(pts_2d)))), axis=0)
        point_indices = np.append(point_indices, np.asarray([i for i in range(len(pts_2d))]), axis=0)
        points_2d = np.vstack((points_2d, np.squeeze(pts_2d, axis=1)))

    points_3d = []
    for i, pts_3d in enumerate(frame_points_3d):
        for pt in pts_3d:
            points_3d.append(pt)
    points_3d = np.asarray(points_3d)

    return camera_params, camera_indices, point_indices, points_3d, points_2d


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
        points3d.append(params[n_cameras * 9 + range_counter:n_cameras * 9 + range_counter + range_points].reshape((range, 1, 3)))
        range_counter += range_points

    return cameras, points3d


def start_bundle_adjustment(cameras, points3d, points2d):
    print('start bundle adjustment ...')
    camera_params, camera_indices, point_indices, points_3d, points_2d = prepare_data(cameras, points3d, points2d)

    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]
    n_points_per_frame = [points.shape[0] for points in points2d]

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    res = least_squares(get_residuals, x0, jac_sparsity=A, verbose=1, x_scale='jac', ftol=1e-5,
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))

    optimized_cameras, optimized_points_3d = optimized_params(res.x, n_cameras, n_points_per_frame)

    return optimized_cameras, optimized_points_3d
