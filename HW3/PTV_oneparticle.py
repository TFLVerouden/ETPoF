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

#search neighbourhood:
R = 7 #pixels
Rz = 3 #allowable difference in z-coordinates between the two particles

#camera A
xA = np.zeros(0)
zA = np.zeros(0)


#camera B
yB = np.zeros(0)
zB = np.zeros(0)


x3D = np.zeros(0)
y3D = np.zeros(0)
z3DA = np.zeros(0)
z3DB = np.zeros(0)

#contour number

N = 0
nr = 1

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
    
    #find contours imgA
    contours, _ = cv2.findContours(threshA, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgA, contours, -1, (0,255,0), 1)
    
    if i == 200:
        #take only nth contour
        #find center of countour imgA
        for j in contours:
            M = cv2.moments(j)
            if M['m00'] != 0:
                #center point at the start
                x00 = M['m10']/M['m00']
                z00 = M['m01']/M['m00']
                cx = int(M['m10']/M['m00'])
                cz = int(M['m01']/M['m00'])
                if N == nr:
                    break
                N+= 1
    
        #find contours of imgB
        contours, _ = cv2.findContours(threshB, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        for j in contours:
            M = cv2.moments(j)
            if M['m00'] != 0:
                yB = np.append(yB, M['m10']/M['m00'])
                zB = np.append(zB, M['m01']/M['m00'])
                cy = int(M['m10']/M['m00'])
                cz = int(M['m01']/M['m00'])
        #find closest matching z-coordinate
        index = np.argmin(abs(zB - z00))
        
        #find corresponding y-coordinate
        y00 = yB[index]
        
        x3D = np.append(x3D, x00)
        y3D = np.append(y3D, y00)
        z3DA = np.append(z3DA, z00)
        z3DB = np.append(z3DB, zB[index])
        
    else:
        #find center of countour imgA
        for j in contours:
            M = cv2.moments(j)
            if M['m00'] != 0:
                xA = np.append(xA, M['m10']/M['m00'])
                zA = np.append(zA, M['m01']/M['m00'])
                cx = int(M['m10']/M['m00'])
                cz = int(M['m01']/M['m00'])
                #print(k)
                #if k ==2:
                #    cv2.line(imgA, (cx, cz), (cx+2, cz+2), (0, 250, 250), 2)
                #    cv2.line(imgB, (0, cz), (450, cz), (0, 250, 250), 1)
        
        #find contours imgB
        contours, _ = cv2.findContours(threshB, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgB, contours, -1, (0,255,0), 1)
        
        for j in contours:
            M = cv2.moments(j)
            if M['m00'] != 0:
                yB = np.append(yB, M['m10']/M['m00'])
                zB = np.append(zB, M['m01']/M['m00'])
                cy = int(M['m10']/M['m00'])
                cz = int(M['m01']/M['m00'])
                
        #previous coordinates:
        xp = x3D[-1]
        yp = y3D[-1]
        zpA = z3DA[-1]
        zpB = z3DB[-1]
        
        while True:
            index = np.argmin(abs(zA - zpA))
            #print(i, xA[index]-xp, zA[index]-zpA)
            if np.sqrt(abs(xA[index]-xp)**2 + abs(zA[index]-zpA)**2) < R:
                break
            else:
                zA = np.delete(zA, index)
                xA = np.delete(xA, index)
                if not len(zA) > 0:
                    print('AAAAAAA')
        
        z3DA = np.append(z3DA, zA[index])
        x3D = np.append(x3D, xA[index])
        
        while True:
            index = np.argmin(abs(zB - zpB))
            if np.sqrt(abs(yB[index]-yp)**2 + abs(zB[index]-zpB)**2) < R:
                if np.sqrt(abs(zB[index]-z3DA[-1])**2) < Rz:
                    #index = np.argmin(abs(zB - zpB))
                    z3DB = np.append(z3DB, z3DA[-1])
                    y3D = np.append(y3D, yB[index])
                else:
                    #index = np.argmin(abs(zB - zpB))
                    z3DB = np.append(z3DB, zB[index])
                    y3D = np.append(y3D, yB[index])
                
                break
            else:
                zB = np.delete(zB, index)
                yB = np.delete(yB, index)
                if len(zB) == 0:
                    print('AAAAAAA')
                    break
        
        
        
        
    #camera A
    xA = np.zeros(0)
    zA = np.zeros(0)


    #camera B
    yB = np.zeros(0)
    zB = np.zeros(0)
    
    cv2.circle(imgA, (int(x3D[-1]), int(z3DA[-1])), 4, (0, 250, 250), 1)
    cv2.circle(imgB, (int(y3D[-1]), int(z3DB[-1])), 4, (0, 250, 250), 1)
    #cv2.line(imgB, (0, int(z3DB[-1])), (450, int(z3DB[-1])), (255, 250, 250), 1)
    #cv2.line(imgB, (0, int(z3DA[-1])), (450, int(z3DA[-1])), (255, 0, 0), 1)
    '''
    plt.figure(1); plt.clf()
    plt.imshow(imgA[int(z3DA[-1])-50:int(z3DA[-1])+50,:])
    plt.pause(0.01)
    
    plt.figure(2); plt.clf()
    plt.imshow(imgB[int(z3DA[-1])-50:int(z3DA[-1])+50,:])
    plt.pause(0.01)
    '''
    
    


'''
plt.figure(1)
plt.scatter(xA, yA)

plt.figure(2)
plt.scatter(xB, yB)
'''

# Creating figure
fig = plt.figure()
ax = plt.axes(projection ="3d")
ax.set_box_aspect((450, 450, 860))
col = np.arange(len(x3D))
cm = plt.get_cmap("RdYlGn")

# Creating plot
ax.plot(x3D, y3D, z3DA, color = "green", alpha = 0.5)
#ax.scatter3D(x3D, y3D, z3DB, color = "blue", alpha = 0.5, s = 10)
plt.title("simple 3D scatter plot")
ax.set_xlabel('X-axis', fontweight ='bold') 
ax.set_ylabel('Y-axis', fontweight ='bold') 
ax.set_zlabel('Z-axis', fontweight ='bold')
ax.set_xlim(0, 450)
ax.set_ylim(0, 450)
ax.set_zlim(0, 860)
# show plot
plt.show()