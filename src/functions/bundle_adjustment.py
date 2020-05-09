from __future__ import print_function
import urllib
import bz2
import os
from itertools import repeat

import cv2
import numpy as np
import time
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt

from functions import *
from util import *

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
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
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

def buildCamera(R, t, K, dist):
    """
    First 3 components in each row form a rotation vector,
    next 3 components form a translation vector,
    then a focal distance and two distortion parameters.
    """
    camera = R.flatten()
    camera = np.append(camera, t.flatten())
    camera = np.append(camera, K[0][0])
    camera = np.append(camera, dist[0, 0:2].flatten())  # TODO our dist in 2 parameter?
    #print(camera.shape)
    return camera

def prepareData(cameras, points3d, points2d):
    camera_params = np.empty((len(cameras), 9))
    for c in cameras:
        R, _ = cv2.Rodrigues(c.R)
        camera = buildCamera(R, c.t, K, dist)
        camera_params = np.append(camera_params, [camera], axis=0)

    camera_indices = np.empty((len(points2d.flatten)))
    point_indices = np.empty((len(points2d.flatten)))
    points_2d = np.empty((len(points2d.flatten), 2))
    for i, pts_2d in enumerate(points2d):
        camera_indices = np.append(camera_indices, [np.asarray(list(repeat(i, len(pts_2d.flatten()))))], axis=0)
        point_indices = np.append(point_indices, [np.asarray([i for i in range(len(pts_2d.flatten()))])], axis=0)
        points_2d = np.vstack(points_2d, np.squeeze(pts_2d, axis=1))

    points_3d = np.squeeze(points3d, axis=1)

    return camera_params, camera_indices, point_indices, points_3d, points_2d

if __name__== "__main__":
    camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)

    print(camera_params.shape)
    print(points_3d.shape)
    print(points_2d.shape)
    print(len(camera_indices))
    print(len(point_indices))

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
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
    plt.plot(f0)
    plt.show()

    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    t1 = time.time()

    print("Optimization took {0:.0f} seconds".format(t1 - t0))

    plt.plot(res.fun)
    plt.show()


'''
R0, _ = cv2.Rodrigues(R0)
R2, _ = cv2.Rodrigues(R2)
camera0 = buildCamera(R0, t0, K, dist)
camera2 = buildCamera(R2, t2, K, dist)
camera_params = np.asarray((camera0, camera2))
camera_indices = np.asarray(list(repeat(0, len(points2d_0))) + list(repeat(1, len(points2d_2))))
point_indices = np.asarray([i for i in range(len(points2d_0))] + [i for i in range(len(points2d_0))])
points_3d = np.squeeze(axis, axis=1)
points_2d = np.vstack((np.squeeze(points2d_0, axis=1), np.squeeze(points2d_2, axis=1)))
#point_indices = list(repeat(0, len(keyframe_pts[0][0]))) + list(repeat(1, len(keyframe_pts[0][-1])))
#points_3d = world_coords1
#points_2d = keyframe_pts[0][0] + keyframe_pts[0][-1]

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
f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
plt.plot(f0)
plt.show()

A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

t0 = time.time()
res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
t1 = time.time()

print("Optimization took {0:.0f} seconds".format(t1 - t0))

plt.plot(res.fun)
plt.show()
'''