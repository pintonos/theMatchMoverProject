from util import *

""" Functions to draw points into an image
"""


def draw_axis(img, points):
    points = np.int32(points).reshape(-1, 2)
    img = cv2.line(img, tuple(points[0]), tuple(points[1]), (255, 0, 0), 3)
    img = cv2.line(img, tuple(points[0]), tuple(points[2]), (0, 255, 0), 3)
    img = cv2.line(img, tuple(points[0]), tuple(points[3]), (0, 0, 255), 3)
    return img


def draw(img, points):
    points = np.int32(points).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [points[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(points[i]), tuple(points[j]), (255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [points[4:]], -1, (0, 0, 255), 3)

    return img


def draw_points(img, pts):
    for i, pt in enumerate(pts):
        cv2.circle(img, (pt[0], pt[1]), 3, (0, 0, 255), -1)
        cv2.putText(img, str(i), (pt[0], pt[1]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)


def get_3d_cube_points():
    return np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]])


def get_3d_axis():
    return np.float32([[0.70423656, 0.3138115, 3.00457495], [0.76984241, 0.14629454, 3.27673669],
                       [1.19341674, 0.3301357, 2.89939043], [0.70700564, 0.36619292, 3.03376407]])


def plot_show_img(img, points, title, axis=False):
    if axis:
        draw_axis(img, points)
    else:
        draw(img, points)
    img = cv2.resize(img, DEMO_RESIZE)
    cv2.imshow(title, img)
