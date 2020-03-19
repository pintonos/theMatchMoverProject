#!/usr/bin/python3
import cv2
import numpy as np

# based on https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/

MAX_FEATURES = 1000

def getHomograhpy(im1, im2, save_image=True):
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
  
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
  
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * 0.1)
  matches = matches[:numGoodMatches]
  
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

  # calculate mean and standard deviation
  mean = np.mean(points1, axis=0)
  sd = np.std(points1, axis=0)
  x_lower, x_upper = int(mean[0] - sd[0]), int(mean[0] + sd[0])
  y_lower, y_upper = int(mean[1] - sd[1]), int(mean[1] + sd[1])

  points1_exact = []
  points2_exact = []
  matchesMask = [0 for i in range(len(matches))]

  # remove incorrect matches
  for i, match in enumerate(matches):
    (x, y) = keypoints1[match.queryIdx].pt
    if x_lower < x < x_upper and y_lower < y < y_upper:
      matchesMask[i]=1
      points1_exact.append(keypoints1[match.queryIdx].pt)
      points2_exact.append(keypoints2[match.trainIdx].pt)

  print('got', len(points1_exact), 'matches')

  if save_image:
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None, matchColor=(0,255,0), matchesMask=matchesMask)
    # draw red rectangle of standard deviations
    imMatches = cv2.rectangle(imMatches, (x_lower, y_lower), (x_upper, y_upper), (0, 0, 255), 3)
    cv2.imwrite("../tmp/matches.jpg", imMatches)
  
  # Find homography
  H, _ = cv2.findHomography(np.array(points1_exact), np.array(points2_exact), cv2.RANSAC)

  return H


image_reference = cv2.imread("../data/visual-geometry-reference.png", cv2.IMREAD_COLOR)
image_sample = cv2.imread("../data/visual-geometry-sample.png", cv2.IMREAD_COLOR)

H = getHomograhpy(image_sample, image_reference)

# Print estimated homography
print("Estimated homography matrix H: \n",  H)
  
