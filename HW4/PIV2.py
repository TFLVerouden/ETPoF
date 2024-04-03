# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:19:19 2024

@author: annem
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import cv2
from scipy.optimize import curve_fit
from PIV import PIV
from PIV import twoD_Gaussian
# B2
img1 = cv2.imread('PIVimages/shearflow/shearFlow1a.png', 0)
img2 = cv2.imread('PIVimages/shearflow/shearFlow1b.png', 0)

#img1 = cv2.imread('PIVimages/solidbody/solidBody2a.png', 0)
#img2 = cv2.imread('PIVimages/solidbody/solidBody2b.png', 0)

#img1 = cv2.imread('PIVimages/commercialflow/bestFlow4a.png', 0)
#img2 = cv2.imread('PIVimages/commercialflow/bestFlow4b.png', 0)


nr = 16
shapeB = img1.shape[0] #-1
window_size = int(shapeB/nr)
amp = 2
for i in range(1, nr+1):
    for j in range(1, nr+1):
        x1 = (i-1)*window_size
        x2 = (i)*window_size
        y1 = (j-1)*window_size
        y2 = (j)*window_size
        
        
        imgA = img1[y1:y2, x1:x2]
        imgB = img2[y1:y2, x1:x2]
        
        xdir, ydir = PIV(imgA, imgB)
        x0 = x1+((x2-x1)/2)
        y0 = y1+((y2-y1)/2)
        #print(xdir, ydir)
        plt.figure(1)
        #plt.imshow(img1)
        plt.hlines(x2, 0, shapeB)
        plt.vlines(y2, 0, shapeB)
        plt.arrow(x0, y0, xdir*amp, ydir*amp, length_includes_head=True, head_width=10, head_length=4)
        
        #plt.figure(2)
        #plt.imshow(imgA)
        #plt.figure(3)
        #plt.imshow(imgB)
        

#plt.imshow(img1)