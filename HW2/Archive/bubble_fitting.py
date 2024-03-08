# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 11:58:50 2024

@author: annem
"""
import cv2 # you might need to install 'opencv-python' first
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
from circle_fit import taubinSVD

def func(x, a, b):
    return a*x + b

nN = 275-3#290 #number of images
fps = 0.01
dt = 1/fps
volume = np.zeros(nN+1)
angle = np.zeros(nN+1)
base_radius = np.zeros(nN+1)
t1 = np.linspace(0, len(angle)*dt-dt, len(angle))

#for i in reversed(range(nN+1-3)):
for i in range(nN+1):
    num = f'{i}' #pad number to 0007 shape
    img = cv2.imread(f'C:/Users/annem/Downloads/University folder/Msc/MOD03/Experimental Techniques/ETPoF_24_HW2/Images/EvaporatingDroplet/experiment{num.zfill(4)}.tif')
    img2 = np.copy(img)
    #turn image gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #gaussian blur
    gray = cv2.GaussianBlur(gray,(5,5),0)
    
    #threshold
    _,thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
    
    #find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    points = np.vstack(contours[-1]).squeeze()
    x, y = points[:,0], points[:,1]

    # taking off the edges of the frame it keeps detecting >:(, and the bottom substrate
    mask = (y != 0) & (x != 0) & (x != np.max(x)) & (y != np.max(y)) & (y<668)# & (y>300)
    points = points[mask]
    
    #reshape into a contour
    ctr = np.array(points).reshape((-1,1,2)).astype(np.int32)
    
        
    #find min enclsoing circle and ellipse of whole drop
    (xc, yc), r = cv2.minEnclosingCircle(ctr)
    ellipse = cv2.fitEllipse(ctr)
    
    # for bonus, fit both circle and ellipse
    cv2.ellipse(img2, ellipse, (255,0,0), 3)
    cv2.circle(img2,(int(round(xc)),int(round(yc))), int(round(r)), (0,0,255), 3)
    
    # finding base radius
    mask2 = (y != 0) & (x != 0) & (x != np.max(x)) & (y != np.max(y)) & (y<668) & (y==668-1)
    xbase = x[mask2]
    x1 = np.min(xbase)
    x2 = np.max(xbase)
    
    base_radius[i] = abs(x1-x2)/2
    
    #take half the shape to integrate and find volume
    mask = (y != 0) & (x != 0) & (x != np.max(x)) & (y != np.max(y)) & (y<668) & (x>xc)
    volume[i] = np.pi*np.trapz((x[mask]-xc)**2, y[mask])
    
    #use only lower part to fit ellipse to determine Young's angle
    mask = (y != 0) & (x != 0) & (x != np.max(x)) & (y != np.max(y)) & (y<668) & (y>yc)# & (x>xc)
    
    points = np.vstack(contours[-1]).squeeze()
    points = points[mask]
    ctr = np.array(points).reshape((-1,1,2)).astype(np.int32)
    
    ellipse = cv2.fitEllipse(ctr)
    
    u=ellipse[0][0]         #x-position of the center
    v=ellipse[0][1]         #y-position of the center
    a=ellipse[1][1]/2       #radius on the x-axis
    b=ellipse[1][0]/2      #radius on the y-axis

    t = np.linspace(0, 2*np.pi, 10000)
    
    y = v+b*np.sin(t)
    x = u+a*np.cos(t)
    
    #fit line and determine angle
    mask = (y<668) & (y>650) & (x>xc)
    y = y[mask]
    x = x[mask]
    
    popt, pcov = curve_fit(func, x, y)
    
    y2 = np.round(func(x2, *popt))
    y3 = np.round(func(x2+200, *popt))
    
    angle[i] = np.arctan(abs(popt[0]))*180/np.pi
    
    #draw baseradius, fitted line and ellipse on circle
    cv2.ellipse(img, ellipse, (255,255,0), 3)
    img = cv2.line(img, (x1, 668), (x2, 668), (255,0,0), 3)
    img = cv2.line(img, (x2, int(y2)), (x2+200, int(y3)), (255,0,255), 3)
    
    
    #For showing process
    plt.figure(1); plt.clf()
    plt.imshow(img)
    plt.pause(0.01)
    
    plt.figure(2); plt.clf()
    plt.imshow(img2)
    plt.title('circle=blue, ellipse=red')
    plt.pause(0.01)
    
    #break
    
plt.figure()
plt.plot(t1, volume)
plt.xlabel('t(s)')
plt.ylabel('V (units)')

plt.figure()
plt.plot(t1, angle)
plt.xlabel('t(s)')
plt.ylabel('angle in degrees')

plt.figure()
plt.plot(t1, base_radius)
plt.xlabel('t(s)')
plt.ylabel('base radius r')
