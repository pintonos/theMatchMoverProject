from util import *

""" Extract two different frames from input video

Frames are saved in DATA_PATH
"""

video = cv2.VideoCapture('../' + VIDEO_PATH)
frames_total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

frame_1 = 0
frame_2 = 30

count = 0
success = True
img_1, img_2, img_3 = None, None, None

while success:
    success, img = video.read()
    if frame_1 == count:
        img_1 = img
    if frame_2 == count:
        img_2 = img
        break
    count += 1

if img_1 is None or img_2 is None:
    raise Exception('Unable to find all frames')

cv2.imwrite('../' + DATA_PATH + 'img_' + str(frame_1) + '.jpg', img_1)
cv2.imwrite('../' + DATA_PATH + 'img_' + str(frame_2) + '.jpg', img_2)

cv2.imshow('img_' + str(frame_1), cv2.resize(img_1, DEMO_RESIZE))
cv2.imshow('img_' + str(frame_2), cv2.resize(img_2, DEMO_RESIZE))

cv2.waitKey(0)
cv2.destroyAllWindows()
