from __future__ import print_function
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

  if save_image:
        # Need to draw only good matches, so create a mask
    matchesMask = [0 for i in range(len(points1))]

    # ratio test as per Lowe's paper
    for i, (x, y) in enumerate(points1):
        if x > mean[0]-sd[0] and x < mean[0]+sd[0] and y > mean[1]-sd[1] and y < mean[1]+sd[1]:
            matchesMask[i]=1

    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None, matchColor=(0,255,0), matchesMask=matchesMask)
    pt1 = (int(mean[0]-sd[0]), int(mean[1]-sd[1]))
    pt2 = (int(mean[0]+sd[0]), int(mean[1]+sd[1]))
    imMatches = cv2.rectangle(imMatches, pt1, pt2, (0, 0, 255), 3)
    cv2.imwrite("../data/matches.jpg", imMatches)
  
  # Find homography
  H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

  return H


image_reference = cv2.imread("../data/visual-geometry-reference.png", cv2.IMREAD_COLOR)
image_sample = cv2.imread("../data/visual-geometry-sample.png", cv2.IMREAD_COLOR)

H = getHomograhpy(image_sample, image_reference)

# Print estimated homography
print("Estimated homography matrix H: \n",  H)
  
