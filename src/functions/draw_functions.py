from util import *

""" Functions to draw points into an image
"""


def draw_axis(img, points):
    points = np.int32(points).reshape(-1, 2)
    img = cv2.line(img, tuple(points[0]), tuple(points[1]), (255,0,0), 3)
    img = cv2.line(img, tuple(points[0]), tuple(points[2]), (0,255,0), 3)
    img = cv2.line(img, tuple(points[0]), tuple(points[3]), (0,0,255), 3)
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
    return np.float32([[4.5684504, 5.56244952, -3.50609609], [5.09206892, 3.32229468, -3.74294852],
                       [8.81651054, 4.7006616, -2.50846552], [4.59012777, 5.58675727, -3.05795159]])


def plot_show_img(img, points, title, axis=False):
    if axis:
        draw_axis(img, points)
    else:
        draw(img, points)
    img = cv2.resize(img, DEMO_RESIZE)
    cv2.imshow(title, img)
