from util.config import *

""" Extract three different frames from input video

Frames are saved in DATA_PATH
"""

DEMO_RESIZE = (960, 540)

video = cv2.VideoCapture('../' + VIDEO_PATH)
frames_total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

frame_1 = 0
frame_2 = frames_total // 2 + 4
frame_3 = frames_total - 1

count = 0
success = True
img_1, img_2, img_3 = None, None, None

while success:
    success, img = video.read()
    if frame_1 == count:
        img_1 = img
    if frame_2 == count:
        img_2 = img
    if frame_3 == count:
        img_3 = img
        break
    count += 1

if img_1 is None or img_2 is None or img_3 is None:
    raise Exception('Unable to find all frames')

cv2.imwrite('../' + DATA_PATH + 'img_1.jpg', img_1)
cv2.imwrite('../' + DATA_PATH + 'img_2.jpg', img_2)
cv2.imwrite('../' + DATA_PATH + 'img_3.jpg', img_3)

cv2.imshow('img_1', cv2.resize(img_1, DEMO_RESIZE))
cv2.imshow('img_2', cv2.resize(img_2, DEMO_RESIZE))
cv2.imshow('img_3', cv2.resize(img_3, DEMO_RESIZE))

cv2.waitKey(0)
cv2.destroyAllWindows()
