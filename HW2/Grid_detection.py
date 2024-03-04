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

#erode, dilate = open
n = 3
kernel = np.ones((n,n),np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

img2 = cv2.blur(opening, (5,5), 0)
_,thresh = cv2.threshold(img2, 40, 255, cv2.THRESH_BINARY)

edges = cv2.Canny(thresh,70,255)

plt.figure()
plt.imshow(edges)

plt.figure()
plt.imshow(opening)

plt.figure()
plt.imshow(img2[200:400, 200:400])

plt.figure()
plt.imshow(thresh)

gray = cv2.imread('Images/Calibration_b/20201204-2x-100um.tif', 0)
dst = cv2.cornerHarris(gray,10,3,0.04)
dst1 = np.copy(dst)
ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
dst = np.uint8(dst)
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
_,thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
plt.imshow(dst1)
print(centroids)
'''
plt.imshow(thresh)
ret, corners = cv2.findChessboardCorners(thresh, (4,4), None)

print(corners)
'''
'''
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
'''