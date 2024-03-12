# -*- coding: utf-8 -*-
"""
Tracking back to front, because particles are well separated at the end
"""
import cv2 # you might need to install 'opencv-python' first
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit

nN = 200

#camera A
xA = np.zeros(0)
zA = np.zeros(0)


#camera B
yB = np.zeros(0)
zB = np.zeros(0)


x3D = np.zeros(0)
y3D = np.zeros(0)
z3D = np.zeros(0)

for i in tqdm(reversed(range(1, nN+1))):
    imgA = cv2.imread(f'PTV/a{i}.png')
    imgB = cv2.imread(f'PTV/b{i}.png')
    
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    
    # Otsu's thresholding after Gaussian filtering
    n = 3
    blurA = cv2.GaussianBlur(grayA,(n,n),0)
    blurB = cv2.GaussianBlur(grayB,(n,n),0)
    
    #threshold
    _,threshA = cv2.threshold(blurA, 130, 255, cv2.THRESH_BINARY)
    _,threshB = cv2.threshold(blurB, 130, 255, cv2.THRESH_BINARY)
    
    #find contours
    contours, _ = cv2.findContours(threshA, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(imgA, contours, -1, (0,255,0), 1)
    
    #find center of countour imgA
    k = 0
    for j in contours:
        M = cv2.moments(j)
        k = k + 1
        if M['m00'] != 0:
            xA = np.append(xA, M['m10']/M['m00'])
            zA = np.append(zA, M['m01']/M['m00'])
            cx = int(M['m10']/M['m00'])
            cz = int(M['m01']/M['m00'])
            print(k)
            if k ==2:
                cv2.line(imgA, (cx, cz), (cx+2, cz+2), (0, 250, 250), 2)
                cv2.line(imgB, (0, cz), (450, cz), (0, 250, 250), 1)
    
    #find contours
    contours, _ = cv2.findContours(threshB, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    
    for j in contours:
        M = cv2.moments(j)
        if M['m00'] != 0:
            yB = np.append(yB, M['m10']/M['m00'])
            zB = np.append(zB, M['m01']/M['m00'])
            cy = int(M['m10']/M['m00'])
            cz = int(M['m01']/M['m00'])
    
    for l in range(len(xA)):
        z0 = zA[l]
        x0 = xA[l]
        zlist = abs(zB - z0)
        index = np.argmin(zlist)
        y0 = yB[index]
        
        x3D = np.append(x3D, x0)
        y3D = np.append(y3D, y0)
        z3D = np.append(z3D, z0)
        
        
    #camera A
    xA = np.zeros(0)
    zA = np.zeros(0)


    #camera B
    yB = np.zeros(0)
    zB = np.zeros(0)

    
    plt.figure(1); plt.clf()
    plt.imshow(imgA)#[:300,:])
    plt.pause(0.01)
    
    plt.figure(2); plt.clf()
    plt.imshow(imgB)#[:300,:])
    plt.pause(0.01)
    break

'''
plt.figure(1)
plt.scatter(xA, yA)

plt.figure(2)
plt.scatter(xB, yB)
'''

# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
ax.scatter3D(x3D, y3D, z3D, color = "green", alpha = 0.5, s = 10)
plt.title("simple 3D scatter plot")
ax.set_xlabel('X-axis', fontweight ='bold') 
ax.set_ylabel('Y-axis', fontweight ='bold') 
ax.set_zlabel('Z-axis', fontweight ='bold')
 
# show plot
plt.show()