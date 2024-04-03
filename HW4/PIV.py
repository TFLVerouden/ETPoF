# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:34:33 2024

@author: annem
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import cv2
from scipy.optimize import curve_fit

def twoD_Gaussian(xy, amp, xo, yo, a, b, c):
    x, y = xy
    inner = a * (x - xo)**2
    inner += 2 * b * (x - xo)**2 * (y - yo)**2
    inner += c * (y - yo)**2
    return amp * np.exp(-inner)

def PIV(img1, img2):
    cor = fftconvolve(img2, img1[::-1,::-1])

    #coordinates of maximum, pixel
    x0 = np.where(cor == np.max(cor))[1][0]
    y0 = np.where(cor == np.max(cor))[0][0]

    # take area of pixels around max, which is 9 datapoints total
    x_arr = np.array([x0-1, x0, x0+1])
    y_arr = np.array([y0-1, y0, y0+1])

    x = np.zeros(0)
    y = np.zeros(0)
    data = np.zeros(0)

    for i in x_arr:
        for j in y_arr:
            x = np.append(x, i)
            y = np.append(y, j)
            data = np.append(data, cor[int(j), int(i)])
    try:
        popt, pcov = curve_fit(twoD_Gaussian, (x, y), data, p0=(1, x0, y0, 1, 1, 1))
        xsub = popt[1]
        ysub = popt[2]
    except:
        xsub = x0
        ysub = y0
    #center of corr image
    shape1 = cor.shape
    cx = (shape1[0]-1)/2
    cy = (shape1[1]-1)/2

    #directions of arrow
    xdir = xsub - cx
    ydir = ysub - cy
    return xdir, ydir


# B1
img1 = cv2.imread('PIVimages/crosscorrelation/crosscorrelate32x32a.png', 0)
img2 = cv2.imread('PIVimages/crosscorrelation/crosscorrelate32x32b.png', 0)


cor = fftconvolve(img2, img1[::-1,::-1])

#coordinates of maximum, pixel
x0 = np.where(cor == np.max(cor))[1][0]
y0 = np.where(cor == np.max(cor))[0][0]

# take area of pixels around max, which is 9 datapoints total
x_arr = np.array([x0-1, x0, x0+1])
y_arr = np.array([y0-1, y0, y0+1])

x = np.zeros(0)
y = np.zeros(0)
data = np.zeros(0)

for i in x_arr:
    for j in y_arr:
        x = np.append(x, i)
        y = np.append(y, j)
        data = np.append(data, cor[int(j), int(i)])

#stolen from the web
def twoD_Gaussian(xy, amp, xo, yo, a, b, c):
    x, y = xy
    inner = a * (x - xo)**2
    inner += 2 * b * (x - xo)**2 * (y - yo)**2
    inner += c * (y - yo)**2
    return amp * np.exp(-inner)


popt, pcov = curve_fit(twoD_Gaussian, (x, y), data, p0=(1, x0, y0, 1, 1, 1))
xsub = popt[1]
ysub = popt[2]

#center of corr image
shape1 = cor.shape
cx = (shape1[0]-1)/2
cy = (shape1[1]-1)/2

#directions of arrow
xdir = xsub - cx
ydir = ysub - cy

#center of image
shape2 = img1.shape
cx2 = (shape2[0]-1)/2
cy2 = (shape2[1]-1)/2

plt.figure()
plt.imshow(cor)
plt.scatter(x0, y0, label='pixel')
plt.scatter(xsub, ysub, label='sub-pixel')
plt.ylim(y0-1.5, y0+1.5)
plt.xlim(x0-1.5, x0+1.5)
plt.legend()

plt.figure()
plt.imshow(cor)
plt.scatter(cx, cy)
plt.scatter(xsub, ysub)
plt.quiver(cx, cy, xdir, ydir)
