import cv2
from src.Constants import *

video_cap = cv2.VideoCapture(VIDEO_PATH)
number_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

img_index1 = 0
img_index2 = number_frames - 1
img_index3 = number_frames // 2 + 4

frame_counter = 0
success = True
img1, img2, img3 = 0, 0, 0
while success:
    img = video_cap.read()
    if img_index1 == frame_counter:
        success, img1 = img
    if img_index3 == frame_counter:
        success, img3 = img
    if img_index2 == frame_counter:
        success, img2 = img
        break
    frame_counter += 1

     
cv2.imwrite(IMAGE_PATH + 'img1.jpg', img1)
cv2.imwrite(IMAGE_PATH + 'img2.jpg', img2)
cv2.imwrite(IMAGE_PATH + 'img3.jpg', img3)

img1 = cv2.resize(img1, (960, 540))
img2 = cv2.resize(img2, (960, 540))
img3 = cv2.resize(img3, (960, 540))
cv2.imshow('image 1', img1)
cv2.imshow('image 2', img2)
cv2.imshow('image 3', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
