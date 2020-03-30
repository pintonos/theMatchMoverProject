import cv2
from Constants import *

video_cap = cv2.VideoCapture(VIDEO_PATH)
number_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

img_index1 = 0
img_index2 = number_frames - 1

frame_counter = 0
success = True
img1, img2 = 0, 0
while success:
    img = video_cap.read()
    if img_index1 == frame_counter:
        success, img1 = img
    if img_index2 == frame_counter:
        success, img2 = img
        break
    frame_counter += 1

cv2.imwrite(IMAGE_PATH + 'img1.jpg', img1)
cv2.imwrite(IMAGE_PATH + 'img2.jpg', img2)
