# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 21:01:29 2024

@author: annem
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import math

img = cv2.imread('Images/Calibration_a/Distorted.png', 0)#[600:800, 400:500]
imgc = cv2.imread('Images/Calibration_a/Distorted.png')
ret,thresh1 = cv2.threshold(img,70,255,cv2.THRESH_BINARY)
'''
# Display image
plt.imshow(img)


#Set Parameters for SimpleBlobDetector
blobParams = cv2.SimpleBlobDetector_Params()

# Filter by Area.
blobParams.filterByArea = True
blobParams.minArea = 10    # minArea may be adjusted to suit for your experiment
blobParams.maxArea = 20000  # maxArea may be adjusted to suit for your experiment


# Filter by Convexity
blobParams.filterByConvexity = False
#blobParams.minConvexity = 0.95

#Create a detector with the parameters
blobDetector = cv2.SimpleBlobDetector_create(blobParams)

keypoints = blobDetector.detect(thresh1)

im_with_keypoints = cv2.drawKeypoints(thresh1, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure()
plt.imshow(im_with_keypoints)
plt.show()


#edges = cv.Canny(img2,50,150,apertureSize = 3)
rows = img.shape[0]
circles = cv2.HoughCircles(cv2.blur(img, (5,5)), cv2.HOUGH_GRADIENT, 1, rows / 8,
                           param1=140, param2=1,
                           minRadius=1, maxRadius=30)
img2 = np.copy(img)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv2.circle(img, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv2.circle(img, center, radius, (0, 255, 0), 3)
plt.figure()
plt.imshow(img)
'''
img2 = cv2.blur(img, (5,5))

edges = cv2.Canny(img2,70,255)

plt.figure()
plt.imshow(edges)

rows = img.shape[0]
circles = cv2.HoughCircles(cv2.blur(img, (5,5)), cv2.HOUGH_GRADIENT, 1, rows / 8,
                           param1=140, param2=1,
                           minRadius=3, maxRadius=14)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv2.circle(img, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv2.circle(img, center, radius, (255, 0, 0), 3)

plt.figure()
plt.imshow(img)
#img2 = cv2.blur(img, (5,5))

#edges = cv2.Canny(img2,70,255)

plt.figure()
plt.imshow(img2)

ret, centers = cv2.findCirclesGrid(img2, (25,25), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
print(ret)
plt.imshow(cv2.drawChessboardCorners(imgc, (25,25), centers, ret))

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((25*25,3), np.float32)
objp[:,:2] = np.mgrid[0:25,0:25].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

objpoints.append(objp)
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners2 = cv2.cornerSubPix(img2,centers, (25,25), (-1,-1), criteria)
imgpoints.append(corners2)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img2.shape[::-1], None, None)
h,  w = img2.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
plt.figure()
plt.imshow(dst)

x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
plt.figure()
plt.imshow(dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )


#ipympl %matplotlib widget