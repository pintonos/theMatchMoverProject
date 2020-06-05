from itertools import repeat
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import time
import urllib
import bz2
import os

from functions import *
from util import *

"""
bundle adjustments, from:
https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
"""

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
FILE_NAME = "problem-49-7776-pre.txt.bz2"
URL = BASE_URL + FILE_NAME

if not os.path.isfile(FILE_NAME):
    urllib.request.urlretrieve(URL, FILE_NAME)

def read_bal_data(file_name):
    with bz2.open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]

        camera_params = np.empty(n_cameras * 9)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))

    return camera_params, points_3d, camera_indices, point_indices, points_2d


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


def prepare_data(cameras, frame_points_3d, frame_points_2d, keyframe_idx):
    camera_params = np.empty((0, 9))
    for c in cameras:
        R, _ = cv2.Rodrigues(c.R_mat)
        camera = build_camera(R, c.t)
        camera_params = np.append(camera_params, [camera], axis=0)

    camera_indices = []
    point_indices = []
    points_2d = np.empty((0, 2))

    camera_id = 0
    pt_id_counter = 0
    last_id = 0
    for keyframe_id in keyframe_idx[1:]:
        pts_2d = frame_points_2d[0]
        for i in range(last_id, keyframe_id):
            pts_2d = frame_points_2d[i]
            for j in range(pts_2d.shape[0]):
                points_2d = np.vstack((points_2d, pts_2d[j]))

            camera_indices += [camera_id for _ in range(len(pts_2d))]
            point_indices += [i for i in range(pt_id_counter, pt_id_counter + len(pts_2d))]

            camera_id += 1
        last_id = keyframe_id
        pt_id_counter = pt_id_counter + len(pts_2d)

    points_3d = np.empty((0, 3))
    for pts_3d in frame_points_3d:
        for j in range(pts_3d.shape[0]):
            points_3d = np.vstack((points_3d, pts_3d[j]))

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
        points3d.append(params[n_cameras * 9 + range_counter:n_cameras * 9 + range_counter + range_points].reshape((range, 1, 3)))
        range_counter += range_points

    return cameras, points3d


# TODO adjust threshold and atol values
def filter_outliers(cameras, points_2d, points_3d):
    in_points_2d = []
    in_points_3d = []

    for i in range(0, len(cameras)):

        reprojected, _ = cv2.projectPoints(np.asarray(points_3d[i]), cameras[i].R_vec, cameras[i].t, K, dist)
        reprojected = np.reshape(reprojected, (len(reprojected), 2))

        close_arr = np.isclose(points_2d[i], reprojected, atol=2)

        frame_in_pts_2d = []
        frame_in_pts_3d = []
        for j in range(0, len(close_arr)):
            if close_arr[j][0] and close_arr[j][1]:
                frame_in_pts_2d.append(points_2d[i][j])
                frame_in_pts_3d.append(points_3d[i][j])

        in_points_2d.append(np.asarray(frame_in_pts_2d))
        in_points_3d.append(np.asarray(frame_in_pts_3d))

    return in_points_3d, in_points_2d


def start_bundle_adjustment(cameras, points3d, points2d, keyframe_idx):
    print('start bundle adjustment ...')
    #points3d, points2d = filter_outliers(cameras, points2d, points3d)
    camera_params, camera_indices, point_indices, points_3d, points_2d = prepare_data(cameras, points3d, points2d, keyframe_idx)

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
    res = least_squares(get_residuals, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    t1 = time.time()

    print("Optimization took {0:.0f} seconds".format(t1 - t0))

    plt.plot(res.fun)
    plt.show()

    optimized_cameras, optimized_points_3d = optimized_params(res.x, n_cameras, n_points_per_frame)

    return optimized_cameras, optimized_points_3d


if __name__ == "__main__":
    camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)

    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]
    n = 9 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]
    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    print(x0.shape)

    f0 = get_residuals(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
    plt.plot(f0)
    plt.show()

    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    t0 = time.time()
    res = least_squares(get_residuals, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    t1 = time.time()

    print("Optimization took {0:.0f} seconds".format(t1 - t0))

    plt.plot(res.fun)
    plt.show()

    print(res.x)