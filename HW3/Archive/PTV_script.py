import numpy as np
from tqdm import tqdm
from scipy import spatial


def PTV_2DMiddle_delete(x, z, middle_frame, n_velocity=4, Rmax_v=3, Rmax_n=3, nN=200):
    # starting points, take points on the last frame, where particles are nicely separated:
    mask0 = (x[:,1] == middle_frame)
    x00 = x[:,0][mask0]
    z00 = z[:,0][mask0]
    
    path_x_back = np.zeros((1,3))
    path_z_back = np.zeros((1,3))
    
    #first, go backwards (aka, reverse)
    for i in (range(len(x00))):
        #save first coordinate
        path_x_back = np.vstack((path_x_back, np.array([x00[i], middle_frame, i])))
        path_z_back = np.vstack((path_z_back, np.array([z00[i], middle_frame, i])))

        nearest = 0
        for j in reversed(range(1, middle_frame)):
            found = False
            # use nearest neighbour for first couple of points
            if nearest <= n_velocity+1:
                # find points on next frame
                mask0 = (x[:,1] == j)
                x001 = x[:,0][mask0]
                z001 = z[:,0][mask0]
        
                #first find closest point to previous position on camera:
                distance = np.sqrt(abs(z001-path_z_back[-1,0])**2 + abs(x001-path_x_back[-1,0])**2)
                index = np.argmin(distance)
                
                if distance[index] < Rmax_n*abs(path_z_back[-1,1]-j):
                    path_x_back = np.vstack((path_x_back, np.array([x001[index], j, i])))
                    path_z_back = np.vstack((path_z_back, np.array([z001[index], j, i])))
                    found = True
                
                #path_x = np.vstack((path_x, np.array([x001[index], j, i])))
                #path_z = np.vstack((path_z, np.array([z001[index], j, i])))
                
                nearest += 1
            else:
                R = 1
                vz = (path_z_back[-1-n_velocity, 0] - path_z_back[-1, 0])/(path_z_back[-1-n_velocity, 1] - path_z_back[-1, 1]) #averaged over n-1 frames
                vx = (path_x_back[-1-n_velocity, 0] - path_x_back[-1, 0])/(path_x_back[-1-n_velocity, 1] - path_x_back[-1, 1]) #averaged over n-1 frames
                
                #calculate search neighbourhood
                neighbourhood_z = path_z_back[-1,0] + vz
                neighbourhood_x = path_x_back[-1,0] + vx
                
                # find points on next frame within neighbourhood
                while True:
                    mask0 = (x[:,1] == j) &  (x[:,0] < (neighbourhood_x + R)) & (x[:,0] > (neighbourhood_x - R)) & (z[:,0] < (neighbourhood_z + R)) & (z[:,0] > (neighbourhood_z - R))
                    x001 = x[:,0][mask0]
                    z001 = z[:,0][mask0]
                    
                    if R > Rmax_n: #use nearest neighbour instead
                        # find points on next frame
                        mask0 = (x[:,1] == j)
                        x001 = x[:,0][mask0]
                        z001 = z[:,0][mask0]
                
                        #first find closest point to previous position on camera:
                        distance = np.sqrt(abs(z001-path_z_back[-1,0])**2 + abs(x001-path_x_back[-1,0])**2)
                        index = np.argmin(distance)
                        
                        #this works for some, not others >:[
                        if distance[index] < Rmax_v*abs(path_z_back[-1,1]-j):
                            path_x_back = np.vstack((path_x_back, np.array([x001[index], j, i])))
                            path_z_back = np.vstack((path_z_back, np.array([z001[index], j, i])))
                            found = True
                        break
                    
                    if len(x001) >= 1:
                        #first find closest point to previous position on camera A:
                        distance = np.sqrt(abs(z001-neighbourhood_z)**2 + abs(x001-neighbourhood_x)**2)
                        index = np.argmin(distance)
                        
                        #if distance[index] < 4*abs(path_z[-1,1]-j):
                        
                        path_x_back = np.vstack((path_x_back, np.array([x001[index], j, i])))
                        path_z_back = np.vstack((path_z_back, np.array([z001[index], j, i])))
                        found = True
                        break
                    R += 1
                #if a particle was matched, take matched particle out
                #if all particles should be visible on that frame
                if len(x001)==66 and found:
                    del_i = np.where(x == path_x_back[-1,:2])[0][0]
                    z = np.delete(z,del_i,0)
                    x = np.delete(x,del_i,0)

    path_x_forw = np.zeros((1,3))
    path_z_forw = np.zeros((1,3))
    
    #second, go forwards
    for i in (range(len(x00))):
        #save first coordinate
        path_x_forw = np.vstack((path_x_forw, np.array([x00[i], middle_frame, i])))
        path_z_forw = np.vstack((path_z_forw, np.array([z00[i], middle_frame, i])))

        nearest = 0
        for j in (range(middle_frame, nN+1)):
            found = False
            # use nearest neighbour for first couple of points
            if nearest <= n_velocity+1:
                # find points on next frame
                mask0 = (x[:,1] == j)
                x001 = x[:,0][mask0]
                z001 = z[:,0][mask0]
        
                #first find closest point to previous position on camera:
                distance = np.sqrt(abs(z001-path_z_forw[-1,0])**2 + abs(x001-path_x_forw[-1,0])**2)
                index = np.argmin(distance)
                
                if distance[index] < Rmax_n*abs(path_z_forw[-1,1]-j):
                    path_x_forw = np.vstack((path_x_forw, np.array([x001[index], j, i])))
                    path_z_forw = np.vstack((path_z_forw, np.array([z001[index], j, i])))
                    found = True
                
                #path_x = np.vstack((path_x, np.array([x001[index], j, i])))
                #path_z = np.vstack((path_z, np.array([z001[index], j, i])))
                
                nearest += 1
            else:
                R = 1
                vz = (path_z_forw[-1-n_velocity, 0] - path_z_forw[-1, 0])/(path_z_forw[-1-n_velocity, 1] - path_z_forw[-1, 1]) #averaged over n-1 frames
                vx = (path_x_forw[-1-n_velocity, 0] - path_x_forw[-1, 0])/(path_x_forw[-1-n_velocity, 1] - path_x_forw[-1, 1]) #averaged over n-1 frames
                
                #calculate search neighbourhood
                neighbourhood_z = path_z_forw[-1,0] + vz
                neighbourhood_x = path_x_forw[-1,0] + vx
                
                # find points on next frame within neighbourhood
                while True:
                    mask0 = (x[:,1] == j) &  (x[:,0] < (neighbourhood_x + R)) & (x[:,0] > (neighbourhood_x - R)) & (z[:,0] < (neighbourhood_z + R)) & (z[:,0] > (neighbourhood_z - R))
                    x001 = x[:,0][mask0]
                    z001 = z[:,0][mask0]
                    
                    if R > Rmax_n: #use nearest neighbour instead
                        # find points on next frame
                        mask0 = (x[:,1] == j)
                        x001 = x[:,0][mask0]
                        z001 = z[:,0][mask0]
                
                        #first find closest point to previous position on camera:
                        distance = np.sqrt(abs(z001-path_z_forw[-1,0])**2 + abs(x001-path_x_forw[-1,0])**2)
                        index = np.argmin(distance)
                        
                        #this works for some, not others >:[
                        if distance[index] < Rmax_v*abs(path_z_forw[-1,1]-j):
                            path_x_forw = np.vstack((path_x_forw, np.array([x001[index], j, i])))
                            path_z_forw = np.vstack((path_z_forw, np.array([z001[index], j, i])))
                            found = True
                        
                        break
                    
                    if len(x001) >= 1:
                        #first find closest point to previous position on camera A:
                        distance = np.sqrt(abs(z001-neighbourhood_z)**2 + abs(x001-neighbourhood_x)**2)
                        index = np.argmin(distance)
                        
                        #if distance[index] < 4*abs(path_z[-1,1]-j):
                        
                        path_x_forw = np.vstack((path_x_forw, np.array([x001[index], j, i])))
                        path_z_forw = np.vstack((path_z_forw, np.array([z001[index], j, i])))
                        found = True
                        break
                #if a particle was matched, take matched particle out
                #if all particles should be visible on that frame
                if len(x001)==66 and found:
                    del_i = np.where(x == path_x_back[-1,:2])[0][0]
                    z = np.delete(z,del_i,0)
                    x = np.delete(x,del_i,0)
                    R += 1
    #merge the two arrays
    #particle numbers should be the same
    
    path_x = np.zeros((1,3))
    path_z = np.zeros((1,3))
    
    for i in range(len(x00)):
        for j in range(1, nN+1):
            mask_forw = (path_x_forw[:,2] == i) & (path_x_forw[:,1] == j)
            p1x = path_x_forw[mask_forw]
            p1z = path_z_forw[mask_forw]
            mask_back = (path_x_back[:,2] == i) & (path_x_back[:,1] == j)
            p2x = path_x_back[mask_back]
            p2z = path_z_back[mask_back]
            
            path_x = np.vstack((path_x, p2x))
            path_x = np.vstack((path_x, p1x))
            
            path_z = np.vstack((path_z, p2z))
            path_z = np.vstack((path_z, p1z))
    
    return path_x[1:,:], path_z[1:,:]



def PTV_3D_delete(x, z1, y, z2, middle_frame, n_vel=4, R=1, Rz=1, Rv=1, nN=200):
    """
    

    Parameters
    ----------
    x : array
        array with x position and frame number of camera A
    z1 : array
        array with z position and frame number of camera A
    middle_frame : int
        fram eon which all particles are visible.
    n_vel : int, optional
        how many points are used to calculate velocity. The default is 4.
    Rx : float or int, optional
        search neighbourhood. The default is 1.
    Rxv : float or int, optional
        search neighbourhood velocity, set to 0 for nearest neighbour. The default is 1.
    nN : int, optional
        final frame number. The default is 200.

    Returns
    -------
    path3D : array
        3D tracks of particles.

    """
    # starting points, take points on middle frame where particles are all visible:
    mask0 = (x[:,1] == middle_frame)
    x00 = x[:,0][mask0]
    z100 = z1[:,0][mask0]
    
    # for camera B
    mask0 = (y[:,1] == middle_frame)
    y00 = y[:,0][mask0]
    z200 = z2[:,0][mask0]
    
    
    path_x_back = np.zeros((1,3))
    path_z1_back = np.zeros((1,3))
    
    path_y_back = np.zeros((1,3))
    path_z2_back = np.zeros((1,3))
    
    #first, go backwards (aka, reverse)
    for i in tqdm(range(len(x00))):
        #save first coordinate
        path_x_back = np.vstack((path_x_back, np.array([x00[i], middle_frame, i])))
        path_z1_back = np.vstack((path_z1_back, np.array([z100[i], middle_frame, i])))
        
        #match z-coordinates
        #assume 3D matching for first coordinate is correct
        dist_z = abs(z200 - z100[i])
        index_z = np.argmin(dist_z)
        
        path_z2_back = np.vstack((path_z2_back, np.array([z200[index_z], middle_frame, i])))
        path_y_back = np.vstack((path_y_back, np.array([y00[index_z], middle_frame, i])))
        
        nearest = 0
        
        for j in (reversed(range(1, middle_frame))):
            found = False
            # use nearest neighbour for first couple of points
            if nearest <= n_vel+1:
                
                # find points on next frame
                mask0 = (x[:,1] == j)
                x00j = x[:,0][mask0]
                z100j = z1[:,0][mask0]
        
                #first find closest point to previous position on camera:
                z1_prev = path_z1_back[-1,0]
                x_prev = path_x_back[-1,0]
                y_prev = path_y_back[-1,0]
                z2_prev = path_z2_back[-1,0]
                
                #for camera A
                distance = np.sqrt(abs(z100j-z1_prev)**2 + abs(x00j-x_prev)**2)
                index_A = np.argmin(distance)
                
                x_new = x00j[index_A]
                z1_new = z100j[index_A]
                
                
                #try to match z coordinate
                mask0 = (y[:,1] == j)
                y00j = y[:,0][mask0]
                z200j = z2[:,0][mask0]
                
                while len(z200j) != 0:
                    dist_z = abs(z200j - z1_new)
                    index_z = np.argmin(dist_z)
                    
                    z2_new = z200j[index_z]
                    y_new = y00j[index_z]
                    
                    #check if z coordinates are close together
                    if np.sqrt(abs(z1_new-z1_prev)**2 + abs(x_new-x_prev)**2 + abs(y_new-y_prev)**2) < R*abs(path_z1_back[-1,1]-j) and abs(z2_new - z1_new) < Rz*abs(path_z1_back[-1,1]-j):
                    #if abs(z2_new - z1_new) < Rz:
                        #if np.sqrt(abs(z1_new-z1_prev)**2 + abs(x_new-x_prev)**2 + + abs(y_new-y_prev)**2) < R:
                        path_z1_back = np.vstack((path_z1_back, np.array([z1_new, j, i])))
                        path_z2_back = np.vstack((path_z2_back, np.array([z2_new, j, i])))
                        path_x_back = np.vstack((path_x_back, np.array([x_new, j, i])))
                        path_y_back = np.vstack((path_y_back, np.array([y_new, j, i])))
                        nearest +=1
                        found = True
                        #break statement did not work for some reason?                        
                        break
                    else:
                        #get rid of wrong z coordinate
                        z200j = np.delete(z200j, index_z)
                        y00j = np.delete(y00j, index_z)
                        #print(len(y00j))
                        #print('not found')
                #if a particle was matched, take matched particle out
                #if all particles should be visible on that frame
                if len(x00j)==66 and found:
                    del_i = np.where(x == path_x_back[-1,:2])[0][0]
                    z1 = np.delete(z1,del_i,0)
                    x = np.delete(x,del_i,0)
                    
                    del_i = np.where(y == path_y_back[-1,:2])[0][0]
                    z2 = np.delete(z2,del_i,0)
                    y = np.delete(y,del_i,0)
                
                        
            # use predictor
            else:
                # find points on next frame
                mask0 = (x[:,1] == j)
                x00j = x[:,0][mask0]
                z100j = z1[:,0][mask0]
                
                mask0 = (y[:,1] == j)
                y00j = y[:,0][mask0]
                z200j = z2[:,0][mask0]
        
                #find previous points for velocity calculation:
                z1_prev = path_z1_back[-1,0]
                x_prev = path_x_back[-1,0]
                y_prev = path_y_back[-1,0]
                z2_prev = path_z2_back[-1,0]
                
                z1_prev2 = path_z1_back[-1-n_vel,0]
                x_prev2 = path_x_back[-1-n_vel,0]
                y_prev2 = path_y_back[-1-n_vel,0]
                z2_prev2 = path_z2_back[-1-n_vel,0]
                
                dt = path_z1_back[-1-n_vel,1] - path_z1_back[-1,1]
                
                vx = (x_prev2 - x_prev)/dt
                vy = (y_prev2 - y_prev)/dt
                vz = (z1_prev2 - z1_prev)/dt
                
                #calculate search neighbourhood
                n_x = x_prev + vx
                n_y = y_prev + vy
                n_z1 = z1_prev + vz
                
                mask_xz = np.sqrt(abs(z100j-n_z1)**2 + abs(x00j-n_x)**2) <= Rv
                mask_yz = np.sqrt(abs(z200j-n_z1)**2 + abs(y00j-n_y)**2) <= Rv
                
                xs = x00j[mask_xz]
                z1s = z100j[mask_xz]
                
                ys = y00j[mask_yz]
                z2s = z200j[mask_yz]
                if len(ys) != 0 and len(xs) != 0: 
                    dist_s = c = abs(z1s[:, None] - z2s[:])
                    ind_i, ind_j = np.unravel_index(np.argmin(dist_s), dist_s.shape)
                    
                    x_new = xs[ind_i]
                    z1_new = z1s[ind_i]
                    
                    y_new = ys[ind_j]
                    z2_new = z2s[ind_j]
                    
                    path_z1_back = np.vstack((path_z1_back, np.array([z1_new, j, i])))
                    path_z2_back = np.vstack((path_z2_back, np.array([z2_new, j, i])))
                    path_x_back = np.vstack((path_x_back, np.array([x_new, j, i])))
                    path_y_back = np.vstack((path_y_back, np.array([y_new, j, i])))
                    found = True
                
                #if not within predictor neighbourhood, use earest neighbour
                else:
                    
                    # find points on next frame
                    mask0 = (x[:,1] == j)
                    x00j = x[:,0][mask0]
                    z100j = z1[:,0][mask0]
            
                    #first find closest point to previous position on camera:
                    z1_prev = path_z1_back[-1,0]
                    x_prev = path_x_back[-1,0]
                    y_prev = path_y_back[-1,0]
                    z2_prev = path_z2_back[-1,0]
                    
                    #for camera A
                    distance = np.sqrt(abs(z100j-z1_prev)**2 + abs(x00j-x_prev)**2)
                    index_A = np.argmin(distance)
                    
                    x_new = x00j[index_A]
                    z1_new = z100j[index_A] 
                    
                    
                    #try to match z coordinate
                    mask0 = (y[:,1] == j)
                    y00j = y[:,0][mask0]
                    z200j = z2[:,0][mask0]
                    
                    while len(z200j) != 0:
                        dist_z = abs(z200j - z1_new)
                        index_z = np.argmin(dist_z)
                        
                        z2_new = z200j[index_z]
                        y_new = y00j[index_z]
                        
                        #check if z coordinates are close together
                        if np.sqrt(abs(z1_new-z1_prev)**2 + abs(x_new-x_prev)**2 + abs(y_new-y_prev)**2) < R*abs(path_z1_back[-1,1]-j) and abs(z2_new - z1_new) < Rz**abs(path_z1_back[-1,1]-j):
                        #if abs(z2_new - z1_new) < Rz:
                            #if np.sqrt(abs(z1_new-z1_prev)**2 + abs(x_new-x_prev)**2 + + abs(y_new-y_prev)**2) < R:
                            path_z1_back = np.vstack((path_z1_back, np.array([z1_new, j, i])))
                            path_z2_back = np.vstack((path_z2_back, np.array([z2_new, j, i])))
                            path_x_back = np.vstack((path_x_back, np.array([x_new, j, i])))
                            path_y_back = np.vstack((path_y_back, np.array([y_new, j, i])))
                            nearest +=1
                            found = True
                            #break statement did not work for some reason?                        
                            break
                        else:
                            #get rid of wrong z coordinate
                            z200j = np.delete(z200j, index_z)
                            y00j = np.delete(y00j, index_z)
                
                #if a particle was matched, take matched particle out
                #if all particles should be visible on that frame
                if len(x00j)==66 and found:
                    del_i = np.where(x == path_x_back[-1,:2])[0][0]
                    z1 = np.delete(z1,del_i,0)
                    x = np.delete(x,del_i,0)
                    
                    del_i = np.where(y == path_y_back[-1,:2])[0][0]
                    z2 = np.delete(z2,del_i,0)
                    y = np.delete(y,del_i,0)
 
    #second, go forward
    path_x_forw = np.zeros((1,3))
    path_z1_forw = np.zeros((1,3))
    #second, go forward
    path_y_forw = np.zeros((1,3))
    path_z2_forw = np.zeros((1,3))
    
    
    for i in tqdm(range(len(x00))):
        #save first coordinate
        path_x_forw = np.vstack((path_x_forw, np.array([x00[i], middle_frame, i])))
        path_z1_forw = np.vstack((path_z1_forw, np.array([z100[i], middle_frame, i])))
        
        #match z-coordinates
        #assume 3D matching for first coordinate is correct
        dist_z = abs(z200 - z100[i])
        index_z = np.argmin(dist_z)
        
        path_z2_forw = np.vstack((path_z2_forw, np.array([z200[index_z], middle_frame, i])))
        path_y_forw = np.vstack((path_y_forw, np.array([y00[index_z], middle_frame, i])))
        
        nearest = 0
        
        for j in ((range(middle_frame, nN+1))):
            
            # use nearest neighbour for first couple of points
            if nearest <= n_vel+1:
                
                # find points on next frame
                mask0 = (x[:,1] == j)
                x00j = x[:,0][mask0]
                z100j = z1[:,0][mask0]
        
                #first find closest point to previous position on camera:
                z1_prev = path_z1_forw[-1,0]
                x_prev = path_x_forw[-1,0]
                y_prev = path_y_forw[-1,0]
                z2_prev = path_z2_forw[-1,0]
                
                #for camera A
                distance = np.sqrt(abs(z100j-z1_prev)**2 + abs(x00j-x_prev)**2)
                index_A = np.argmin(distance)
                
                x_new = x00j[index_A]
                z1_new = z100j[index_A] 
                
                
                #try to match z coordinate
                mask0 = (y[:,1] == j)
                y00j = y[:,0][mask0]
                z200j = z2[:,0][mask0]
                
                while len(z200j) != 0:
                    dist_z = abs(z200j - z1_new)
                    index_z = np.argmin(dist_z)
                    
                    z2_new = z200j[index_z]
                    y_new = y00j[index_z]
                    
                    #check if z coordinates are close together
                    if np.sqrt(abs(z1_new-z1_prev)**2 + abs(x_new-x_prev)**2 + abs(y_new-y_prev)**2) < R*abs(path_z1_forw[-1,1]-j) and abs(z2_new - z1_new) < Rz*abs(path_z1_forw[-1,1]-j):
                    #if abs(z2_new - z1_new) < Rz:
                        #if np.sqrt(abs(z1_new-z1_prev)**2 + abs(x_new-x_prev)**2 + + abs(y_new-y_prev)**2) < R:
                        path_z1_forw = np.vstack((path_z1_forw, np.array([z1_new, j, i])))
                        path_z2_forw = np.vstack((path_z2_forw, np.array([z2_new, j, i])))
                        path_x_forw = np.vstack((path_x_forw, np.array([x_new, j, i])))
                        path_y_forw = np.vstack((path_y_forw, np.array([y_new, j, i])))
                        nearest +=1
                        #break statement did not work for some reason?                        
                        break
                    else:
                        #get rid of wrong z coordinate
                        z200j = np.delete(z200j, index_z)
                        y00j = np.delete(y00j, index_z)
                        #print(len(y00j))
                        #print('not found')
            # use predictor
            else:
                # find points on next frame
                mask0 = (x[:,1] == j)
                x00j = x[:,0][mask0]
                z100j = z1[:,0][mask0]
                
                mask0 = (y[:,1] == j)
                y00j = y[:,0][mask0]
                z200j = z2[:,0][mask0]
        
                #find previous points for velocity calculation:
                z1_prev = path_z1_forw[-1,0]
                x_prev = path_x_forw[-1,0]
                y_prev = path_y_forw[-1,0]
                z2_prev = path_z2_forw[-1,0]
                
                z1_prev2 = path_z1_forw[-1-n_vel,0]
                x_prev2 = path_x_forw[-1-n_vel,0]
                y_prev2 = path_y_forw[-1-n_vel,0]
                z2_prev2 = path_z2_forw[-1-n_vel,0]
                
                dt = path_z1_forw[-1-n_vel,1] - path_z1_forw[-1,1]
                
                vx = (x_prev2 - x_prev)/dt
                vy = (y_prev2 - y_prev)/dt
                vz = (z1_prev2 - z1_prev)/dt
                
                #calculate search neighbourhood
                n_x = x_prev + vx
                n_y = y_prev + vy
                n_z1 = z1_prev + vz
                
                mask_xz = np.sqrt(abs(z100j-n_z1)**2 + abs(x00j-n_x)**2) <= Rv
                mask_yz = np.sqrt(abs(z200j-n_z1)**2 + abs(y00j-n_y)**2) <= Rv
                
                xs = x00j[mask_xz]
                z1s = z100j[mask_xz]
                
                ys = y00j[mask_yz]
                z2s = z200j[mask_yz]
                if len(ys) != 0 and len(xs) != 0: 
                    dist_s = c = abs(z1s[:, None] - z2s[:])
                    ind_i, ind_j = np.unravel_index(np.argmin(dist_s), dist_s.shape)
                    
                    x_new = xs[ind_i]
                    z1_new = z1s[ind_i]
                    
                    y_new = ys[ind_j]
                    z2_new = z2s[ind_j]
                    
                    path_z1_forw = np.vstack((path_z1_forw, np.array([z1_new, j, i])))
                    path_z2_forw = np.vstack((path_z2_forw, np.array([z2_new, j, i])))
                    path_x_forw = np.vstack((path_x_forw, np.array([x_new, j, i])))
                    path_y_forw = np.vstack((path_y_forw, np.array([y_new, j, i])))
                
                #if not within predictor neighbourhood, use earest neighbour
                else:
                    
                    # find points on next frame
                    mask0 = (x[:,1] == j)
                    x00j = x[:,0][mask0]
                    z100j = z1[:,0][mask0]
            
                    #first find closest point to previous position on camera:
                    z1_prev = path_z1_forw[-1,0]
                    x_prev = path_x_forw[-1,0]
                    y_prev = path_y_forw[-1,0]
                    z2_prev = path_z2_forw[-1,0]
                    
                    #for camera A
                    distance = np.sqrt(abs(z100j-z1_prev)**2 + abs(x00j-x_prev)**2)
                    index_A = np.argmin(distance)
                    
                    x_new = x00j[index_A]
                    z1_new = z100j[index_A] 
                    
                    
                    #try to match z coordinate
                    mask0 = (y[:,1] == j)
                    y00j = y[:,0][mask0]
                    z200j = z2[:,0][mask0]
                    
                    while len(z200j) != 0:
                        dist_z = abs(z200j - z1_new)
                        index_z = np.argmin(dist_z)
                        
                        z2_new = z200j[index_z]
                        y_new = y00j[index_z]
                        
                        #check if z coordinates are close together
                        if np.sqrt(abs(z1_new-z1_prev)**2 + abs(x_new-x_prev)**2 + abs(y_new-y_prev)**2) < R*abs(path_z1_forw[-1,1]-j) and abs(z2_new - z1_new) < Rz*abs(path_z1_forw[-1,1]-j):
                        #if abs(z2_new - z1_new) < Rz:
                            #if np.sqrt(abs(z1_new-z1_prev)**2 + abs(x_new-x_prev)**2 + + abs(y_new-y_prev)**2) < R:
                            path_z1_forw = np.vstack((path_z1_forw, np.array([z1_new, j, i])))
                            path_z2_forw = np.vstack((path_z2_forw, np.array([z2_new, j, i])))
                            path_x_forw = np.vstack((path_x_forw, np.array([x_new, j, i])))
                            path_y_forw = np.vstack((path_y_forw, np.array([y_new, j, i])))
                            nearest +=1
                            #break statement did not work for some reason?                        
                            break
                        else:
                            #get rid of wrong z coordinate
                            z200j = np.delete(z200j, index_z)
                            y00j = np.delete(y00j, index_z) 
                            
    
    path_x = np.zeros((1,3))
    path_y = np.zeros((1,3))
    path_z = np.zeros((1,3))
    
    for i in range(len(x00)):
        for j in range(1, nN+1):
            mask_forw = (path_x_forw[:,2] == i) & (path_x_forw[:,1] == j)
            p1x = path_x_forw[mask_forw]
            p1y = path_y_forw[mask_forw]
            p1z = path_z1_forw[mask_forw]
            mask_back = (path_x_back[:,2] == i) & (path_x_back[:,1] == j)
            p2x = path_x_back[mask_back]
            p2y = path_y_back[mask_back]
            p2z = path_z1_back[mask_back]
            
            path_x = np.vstack((path_x, p2x))
            path_x = np.vstack((path_x, p1x))
            
            path_y = np.vstack((path_y, p2y))
            path_y = np.vstack((path_y, p1y))
            
            path_z = np.vstack((path_z, p2z))
            path_z = np.vstack((path_z, p1z))
    
    return path_x[1:,:], path_y[1:,:], path_z[1:,:]

def PTV_3D(x, z1, y, z2, middle_frame, n_vel=4, R=1, Rz=1, Rv=1, nN=200):
    """
    

    Parameters
    ----------
    x : array
        array with x position and frame number of camera A
    z1 : array
        array with z position and frame number of camera A
    middle_frame : int
        fram eon which all particles are visible.
    n_vel : int, optional
        how many points are used to calculate velocity. The default is 4.
    Rx : float or int, optional
        search neighbourhood. The default is 1.
    Rxv : float or int, optional
        search neighbourhood velocity, set to 0 for nearest neighbour. The default is 1.
    nN : int, optional
        final frame number. The default is 200.

    Returns
    -------
    path3D : array
        3D tracks of particles.

    """
    # starting points, take points on middle frame where particles are all visible:
    mask0 = (x[:,1] == middle_frame)
    x00 = x[:,0][mask0]
    z100 = z1[:,0][mask0]
    
    # for camera B
    mask0 = (y[:,1] == middle_frame)
    y00 = y[:,0][mask0]
    z200 = z2[:,0][mask0]
    
    
    path_x_back = np.zeros((1,3))
    path_z1_back = np.zeros((1,3))
    
    path_y_back = np.zeros((1,3))
    path_z2_back = np.zeros((1,3))
    
    #first, go backwards (aka, reverse)
    for i in tqdm(range(len(x00))):
        #save first coordinate
        path_x_back = np.vstack((path_x_back, np.array([x00[i], middle_frame, i])))
        path_z1_back = np.vstack((path_z1_back, np.array([z100[i], middle_frame, i])))
        
        #match z-coordinates
        #assume 3D matching for first coordinate is correct
        dist_z = abs(z200 - z100[i])
        index_z = np.argmin(dist_z)
        
        path_z2_back = np.vstack((path_z2_back, np.array([z200[index_z], middle_frame, i])))
        path_y_back = np.vstack((path_y_back, np.array([y00[index_z], middle_frame, i])))
        
        nearest = 0
        
        for j in (reversed(range(1, middle_frame))):
            
            # use nearest neighbour for first couple of points
            if nearest <= n_vel+1:
                
                # find points on next frame
                mask0 = (x[:,1] == j)
                x00j = x[:,0][mask0]
                z100j = z1[:,0][mask0]
        
                #first find closest point to previous position on camera:
                z1_prev = path_z1_back[-1,0]
                x_prev = path_x_back[-1,0]
                y_prev = path_y_back[-1,0]
                z2_prev = path_z2_back[-1,0]
                
                #for camera A
                distance = np.sqrt(abs(z100j-z1_prev)**2 + abs(x00j-x_prev)**2)
                index_A = np.argmin(distance)
                
                x_new = x00j[index_A]
                z1_new = z100j[index_A]
                
                
                #try to match z coordinate
                mask0 = (y[:,1] == j)
                y00j = y[:,0][mask0]
                z200j = z2[:,0][mask0]
                
                while len(z200j) != 0:
                    dist_z = abs(z200j - z1_new)
                    index_z = np.argmin(dist_z)
                    
                    z2_new = z200j[index_z]
                    y_new = y00j[index_z]
                    
                    #check if z coordinates are close together
                    if np.sqrt(abs(z1_new-z1_prev)**2 + abs(x_new-x_prev)**2 + abs(y_new-y_prev)**2) < R*abs(path_z1_back[-1,1]-j) and abs(z2_new - z1_new) < Rz*abs(path_z1_back[-1,1]-j):
                    #if abs(z2_new - z1_new) < Rz:
                        #if np.sqrt(abs(z1_new-z1_prev)**2 + abs(x_new-x_prev)**2 + + abs(y_new-y_prev)**2) < R:
                        path_z1_back = np.vstack((path_z1_back, np.array([z1_new, j, i])))
                        path_z2_back = np.vstack((path_z2_back, np.array([z2_new, j, i])))
                        path_x_back = np.vstack((path_x_back, np.array([x_new, j, i])))
                        path_y_back = np.vstack((path_y_back, np.array([y_new, j, i])))
                        nearest +=1
                        #break statement did not work for some reason?                        
                        break
                    else:
                        #get rid of wrong z coordinate
                        z200j = np.delete(z200j, index_z)
                        y00j = np.delete(y00j, index_z)
                        #print(len(y00j))
                        #print('not found')
                
                        
            # use predictor
            else:
                # find points on next frame
                mask0 = (x[:,1] == j)
                x00j = x[:,0][mask0]
                z100j = z1[:,0][mask0]
                
                mask0 = (y[:,1] == j)
                y00j = y[:,0][mask0]
                z200j = z2[:,0][mask0]
        
                #find previous points for velocity calculation:
                z1_prev = path_z1_back[-1,0]
                x_prev = path_x_back[-1,0]
                y_prev = path_y_back[-1,0]
                z2_prev = path_z2_back[-1,0]
                
                z1_prev2 = path_z1_back[-1-n_vel,0]
                x_prev2 = path_x_back[-1-n_vel,0]
                y_prev2 = path_y_back[-1-n_vel,0]
                z2_prev2 = path_z2_back[-1-n_vel,0]
                
                dt = path_z1_back[-1-n_vel,1] - path_z1_back[-1,1]
                
                vx = (x_prev2 - x_prev)/dt
                vy = (y_prev2 - y_prev)/dt
                vz = (z1_prev2 - z1_prev)/dt
                
                #calculate search neighbourhood
                n_x = x_prev + vx
                n_y = y_prev + vy
                n_z1 = z1_prev + vz
                
                mask_xz = np.sqrt(abs(z100j-n_z1)**2 + abs(x00j-n_x)**2) <= Rv
                mask_yz = np.sqrt(abs(z200j-n_z1)**2 + abs(y00j-n_y)**2) <= Rv
                
                xs = x00j[mask_xz]
                z1s = z100j[mask_xz]
                
                ys = y00j[mask_yz]
                z2s = z200j[mask_yz]
                if len(ys) != 0 and len(xs) != 0: 
                    dist_s = c = abs(z1s[:, None] - z2s[:])
                    ind_i, ind_j = np.unravel_index(np.argmin(dist_s), dist_s.shape)
                    
                    x_new = xs[ind_i]
                    z1_new = z1s[ind_i]
                    
                    y_new = ys[ind_j]
                    z2_new = z2s[ind_j]
                    
                    path_z1_back = np.vstack((path_z1_back, np.array([z1_new, j, i])))
                    path_z2_back = np.vstack((path_z2_back, np.array([z2_new, j, i])))
                    path_x_back = np.vstack((path_x_back, np.array([x_new, j, i])))
                    path_y_back = np.vstack((path_y_back, np.array([y_new, j, i])))
                
                #if not within predictor neighbourhood, use earest neighbour
                else:
                    
                    # find points on next frame
                    mask0 = (x[:,1] == j)
                    x00j = x[:,0][mask0]
                    z100j = z1[:,0][mask0]
            
                    #first find closest point to previous position on camera:
                    z1_prev = path_z1_back[-1,0]
                    x_prev = path_x_back[-1,0]
                    y_prev = path_y_back[-1,0]
                    z2_prev = path_z2_back[-1,0]
                    
                    #for camera A
                    distance = np.sqrt(abs(z100j-z1_prev)**2 + abs(x00j-x_prev)**2)
                    index_A = np.argmin(distance)
                    
                    x_new = x00j[index_A]
                    z1_new = z100j[index_A] 
                    
                    
                    #try to match z coordinate
                    mask0 = (y[:,1] == j)
                    y00j = y[:,0][mask0]
                    z200j = z2[:,0][mask0]
                    
                    while len(z200j) != 0:
                        dist_z = abs(z200j - z1_new)
                        index_z = np.argmin(dist_z)
                        
                        z2_new = z200j[index_z]
                        y_new = y00j[index_z]
                        
                        #check if z coordinates are close together
                        if np.sqrt(abs(z1_new-z1_prev)**2 + abs(x_new-x_prev)**2 + abs(y_new-y_prev)**2) < R*abs(path_z1_back[-1,1]-j) and abs(z2_new - z1_new) < Rz**abs(path_z1_back[-1,1]-j):
                        #if abs(z2_new - z1_new) < Rz:
                            #if np.sqrt(abs(z1_new-z1_prev)**2 + abs(x_new-x_prev)**2 + + abs(y_new-y_prev)**2) < R:
                            path_z1_back = np.vstack((path_z1_back, np.array([z1_new, j, i])))
                            path_z2_back = np.vstack((path_z2_back, np.array([z2_new, j, i])))
                            path_x_back = np.vstack((path_x_back, np.array([x_new, j, i])))
                            path_y_back = np.vstack((path_y_back, np.array([y_new, j, i])))
                            nearest +=1
                            #break statement did not work for some reason?                        
                            break
                        else:
                            #get rid of wrong z coordinate
                            z200j = np.delete(z200j, index_z)
                            y00j = np.delete(y00j, index_z)
    #second, go forward
    path_x_forw = np.zeros((1,3))
    path_z1_forw = np.zeros((1,3))
    #second, go forward
    path_y_forw = np.zeros((1,3))
    path_z2_forw = np.zeros((1,3))
    
    
    for i in tqdm(range(len(x00))):
        #save first coordinate
        path_x_forw = np.vstack((path_x_forw, np.array([x00[i], middle_frame, i])))
        path_z1_forw = np.vstack((path_z1_forw, np.array([z100[i], middle_frame, i])))
        
        #match z-coordinates
        #assume 3D matching for first coordinate is correct
        dist_z = abs(z200 - z100[i])
        index_z = np.argmin(dist_z)
        
        path_z2_forw = np.vstack((path_z2_forw, np.array([z200[index_z], middle_frame, i])))
        path_y_forw = np.vstack((path_y_forw, np.array([y00[index_z], middle_frame, i])))
        
        nearest = 0
        
        for j in ((range(middle_frame, nN+1))):
            
            # use nearest neighbour for first couple of points
            if nearest <= n_vel+1:
                
                # find points on next frame
                mask0 = (x[:,1] == j)
                x00j = x[:,0][mask0]
                z100j = z1[:,0][mask0]
        
                #first find closest point to previous position on camera:
                z1_prev = path_z1_forw[-1,0]
                x_prev = path_x_forw[-1,0]
                y_prev = path_y_forw[-1,0]
                z2_prev = path_z2_forw[-1,0]
                
                #for camera A
                distance = np.sqrt(abs(z100j-z1_prev)**2 + abs(x00j-x_prev)**2)
                index_A = np.argmin(distance)
                
                x_new = x00j[index_A]
                z1_new = z100j[index_A] 
                
                
                #try to match z coordinate
                mask0 = (y[:,1] == j)
                y00j = y[:,0][mask0]
                z200j = z2[:,0][mask0]
                
                while len(z200j) != 0:
                    dist_z = abs(z200j - z1_new)
                    index_z = np.argmin(dist_z)
                    
                    z2_new = z200j[index_z]
                    y_new = y00j[index_z]
                    
                    #check if z coordinates are close together
                    if np.sqrt(abs(z1_new-z1_prev)**2 + abs(x_new-x_prev)**2 + abs(y_new-y_prev)**2) < R*abs(path_z1_forw[-1,1]-j) and abs(z2_new - z1_new) < Rz*abs(path_z1_forw[-1,1]-j):
                    #if abs(z2_new - z1_new) < Rz:
                        #if np.sqrt(abs(z1_new-z1_prev)**2 + abs(x_new-x_prev)**2 + + abs(y_new-y_prev)**2) < R:
                        path_z1_forw = np.vstack((path_z1_forw, np.array([z1_new, j, i])))
                        path_z2_forw = np.vstack((path_z2_forw, np.array([z2_new, j, i])))
                        path_x_forw = np.vstack((path_x_forw, np.array([x_new, j, i])))
                        path_y_forw = np.vstack((path_y_forw, np.array([y_new, j, i])))
                        nearest +=1
                        #break statement did not work for some reason?                        
                        break
                    else:
                        #get rid of wrong z coordinate
                        z200j = np.delete(z200j, index_z)
                        y00j = np.delete(y00j, index_z)
                        #print(len(y00j))
                        #print('not found')
            # use predictor
            else:
                # find points on next frame
                mask0 = (x[:,1] == j)
                x00j = x[:,0][mask0]
                z100j = z1[:,0][mask0]
                
                mask0 = (y[:,1] == j)
                y00j = y[:,0][mask0]
                z200j = z2[:,0][mask0]
        
                #find previous points for velocity calculation:
                z1_prev = path_z1_forw[-1,0]
                x_prev = path_x_forw[-1,0]
                y_prev = path_y_forw[-1,0]
                z2_prev = path_z2_forw[-1,0]
                
                z1_prev2 = path_z1_forw[-1-n_vel,0]
                x_prev2 = path_x_forw[-1-n_vel,0]
                y_prev2 = path_y_forw[-1-n_vel,0]
                z2_prev2 = path_z2_forw[-1-n_vel,0]
                
                dt = path_z1_forw[-1-n_vel,1] - path_z1_forw[-1,1]
                
                vx = (x_prev2 - x_prev)/dt
                vy = (y_prev2 - y_prev)/dt
                vz = (z1_prev2 - z1_prev)/dt
                
                #calculate search neighbourhood
                n_x = x_prev + vx
                n_y = y_prev + vy
                n_z1 = z1_prev + vz
                
                mask_xz = np.sqrt(abs(z100j-n_z1)**2 + abs(x00j-n_x)**2) <= Rv
                mask_yz = np.sqrt(abs(z200j-n_z1)**2 + abs(y00j-n_y)**2) <= Rv
                
                xs = x00j[mask_xz]
                z1s = z100j[mask_xz]
                
                ys = y00j[mask_yz]
                z2s = z200j[mask_yz]
                if len(ys) != 0 and len(xs) != 0: 
                    dist_s = c = abs(z1s[:, None] - z2s[:])
                    ind_i, ind_j = np.unravel_index(np.argmin(dist_s), dist_s.shape)
                    
                    x_new = xs[ind_i]
                    z1_new = z1s[ind_i]
                    
                    y_new = ys[ind_j]
                    z2_new = z2s[ind_j]
                    
                    path_z1_forw = np.vstack((path_z1_forw, np.array([z1_new, j, i])))
                    path_z2_forw = np.vstack((path_z2_forw, np.array([z2_new, j, i])))
                    path_x_forw = np.vstack((path_x_forw, np.array([x_new, j, i])))
                    path_y_forw = np.vstack((path_y_forw, np.array([y_new, j, i])))
                
                #if not within predictor neighbourhood, use earest neighbour
                else:
                    
                    # find points on next frame
                    mask0 = (x[:,1] == j)
                    x00j = x[:,0][mask0]
                    z100j = z1[:,0][mask0]
            
                    #first find closest point to previous position on camera:
                    z1_prev = path_z1_forw[-1,0]
                    x_prev = path_x_forw[-1,0]
                    y_prev = path_y_forw[-1,0]
                    z2_prev = path_z2_forw[-1,0]
                    
                    #for camera A
                    distance = np.sqrt(abs(z100j-z1_prev)**2 + abs(x00j-x_prev)**2)
                    index_A = np.argmin(distance)
                    
                    x_new = x00j[index_A]
                    z1_new = z100j[index_A] 
                    
                    
                    #try to match z coordinate
                    mask0 = (y[:,1] == j)
                    y00j = y[:,0][mask0]
                    z200j = z2[:,0][mask0]
                    
                    while len(z200j) != 0:
                        dist_z = abs(z200j - z1_new)
                        index_z = np.argmin(dist_z)
                        
                        z2_new = z200j[index_z]
                        y_new = y00j[index_z]
                        
                        #check if z coordinates are close together
                        if np.sqrt(abs(z1_new-z1_prev)**2 + abs(x_new-x_prev)**2 + abs(y_new-y_prev)**2) < R*abs(path_z1_forw[-1,1]-j) and abs(z2_new - z1_new) < Rz*abs(path_z1_forw[-1,1]-j):
                        #if abs(z2_new - z1_new) < Rz:
                            #if np.sqrt(abs(z1_new-z1_prev)**2 + abs(x_new-x_prev)**2 + + abs(y_new-y_prev)**2) < R:
                            path_z1_forw = np.vstack((path_z1_forw, np.array([z1_new, j, i])))
                            path_z2_forw = np.vstack((path_z2_forw, np.array([z2_new, j, i])))
                            path_x_forw = np.vstack((path_x_forw, np.array([x_new, j, i])))
                            path_y_forw = np.vstack((path_y_forw, np.array([y_new, j, i])))
                            nearest +=1
                            #break statement did not work for some reason?                        
                            break
                        else:
                            #get rid of wrong z coordinate
                            z200j = np.delete(z200j, index_z)
                            y00j = np.delete(y00j, index_z) 
                            
    
    path_x = np.zeros((1,3))
    path_y = np.zeros((1,3))
    path_z = np.zeros((1,3))
    
    for i in range(len(x00)):
        for j in range(1, nN+1):
            mask_forw = (path_x_forw[:,2] == i) & (path_x_forw[:,1] == j)
            p1x = path_x_forw[mask_forw]
            p1y = path_y_forw[mask_forw]
            p1z = path_z1_forw[mask_forw]
            mask_back = (path_x_back[:,2] == i) & (path_x_back[:,1] == j)
            p2x = path_x_back[mask_back]
            p2y = path_y_back[mask_back]
            p2z = path_z1_back[mask_back]
            
            path_x = np.vstack((path_x, p2x))
            path_x = np.vstack((path_x, p1x))
            
            path_y = np.vstack((path_y, p2y))
            path_y = np.vstack((path_y, p1y))
            
            path_z = np.vstack((path_z, p2z))
            path_z = np.vstack((path_z, p1z))
    
    return path_x[1:,:], path_y[1:,:], path_z[1:,:]
                    








def PTV_2D(x, z, backwards=True, n_velocity=4, Rmax_v=3, Rmax_n=3, nN=200):
    if backwards:
        # starting points, take points on the last frame, where particles are nicely separated:
        mask0 = (x[:,1] == nN)
        x00 = x[:,0][mask0]
        z00 = z[:,0][mask0]
        
        path_x = np.zeros((1,3))
        path_z = np.zeros((1,3))
        
        for i in (range(len(x00))):
            #save first coordinate
            path_x = np.vstack((path_x, np.array([x00[i], nN, i])))
            path_z = np.vstack((path_z, np.array([z00[i], nN, i])))
    
            nearest = 0
            for j in reversed(range(1, nN)):
                
                # use nearest neighbour for first couple of points
                if nearest <= n_velocity+1:
                    # find points on next frame
                    mask0 = (x[:,1] == j)
                    x001 = x[:,0][mask0]
                    z001 = z[:,0][mask0]
            
                    #first find closest point to previous position on camera:
                    distance = np.sqrt(abs(z001-path_z[-1,0])**2 + abs(x001-path_x[-1,0])**2)
                    index = np.argmin(distance)
                    
                    if distance[index] < Rmax_n*abs(path_z[-1,1]-j):
                        path_x = np.vstack((path_x, np.array([x001[index], j, i])))
                        path_z = np.vstack((path_z, np.array([z001[index], j, i])))
                    
                    #path_x = np.vstack((path_x, np.array([x001[index], j, i])))
                    #path_z = np.vstack((path_z, np.array([z001[index], j, i])))
                    
                    nearest += 1
                else:
                    R = 1
                    vz = (path_z[-1-n_velocity, 0] - path_z[-1, 0])/(path_z[-1-n_velocity, 1] - path_z[-1, 1]) #averaged over n-1 frames
                    vx = (path_x[-1-n_velocity, 0] - path_x[-1, 0])/(path_x[-1-n_velocity, 1] - path_x[-1, 1]) #averaged over n-1 frames
                    
                    #calculate search neighbourhood
                    neighbourhood_z = path_z[-1,0] + vz
                    neighbourhood_x = path_x[-1,0] + vx
                    
                    # find points on next frame within neighbourhood
                    while True:
                        mask0 = (x[:,1] == j) &  (x[:,0] < (neighbourhood_x + R)) & (x[:,0] > (neighbourhood_x - R)) & (z[:,0] < (neighbourhood_z + R)) & (z[:,0] > (neighbourhood_z - R))
                        x001 = x[:,0][mask0]
                        z001 = z[:,0][mask0]
                        
                        if R > Rmax_n: #use nearest neighbour instead
                            # find points on next frame
                            mask0 = (x[:,1] == j)
                            x001 = x[:,0][mask0]
                            z001 = z[:,0][mask0]
                    
                            #first find closest point to previous position on camera:
                            distance = np.sqrt(abs(z001-path_z[-1,0])**2 + abs(x001-path_x[-1,0])**2)
                            index = np.argmin(distance)
                            
                            #this works for some, not others >:[
                            if distance[index] < Rmax_v*abs(path_z[-1,1]-j):
                                path_x = np.vstack((path_x, np.array([x001[index], j, i])))
                                path_z = np.vstack((path_z, np.array([z001[index], j, i])))
                            
                            break
                        
                        if len(x001) >= 1:
                            #first find closest point to previous position on camera A:
                            distance = np.sqrt(abs(z001-neighbourhood_z)**2 + abs(x001-neighbourhood_x)**2)
                            index = np.argmin(distance)
                            
                            #if distance[index] < 4*abs(path_z[-1,1]-j):
                            
                            path_x = np.vstack((path_x, np.array([x001[index], j, i])))
                            path_z = np.vstack((path_z, np.array([z001[index], j, i])))
                            break
                        R += 1
    else:
        # starting points, take points on the first frame:
        #nN = np.min(x[:,1])
        mask0 = (x[:,1] == 1)
        x00 = x[:,0][mask0]
        z00 = z[:,0][mask0]
        
        path_x = np.zeros((1,3))
        path_z = np.zeros((1,3))
        
        for i in (range(len(x00))):
            #save first coordinate
            path_x = np.vstack((path_x, np.array([x00[i], 1, i])))
            path_z = np.vstack((path_z, np.array([z00[i], 1, i])))
    
            nearest = 0
            for j in (range(1, nN+1)):
                
                # use nearest neighbour for first couple of points
                if nearest <= n_velocity+1:
                    # find points on next frame
                    mask0 = (x[:,1] == j)
                    x001 = x[:,0][mask0]
                    z001 = z[:,0][mask0]
            
                    #first find closest point to previous position on camera:
                    distance = np.sqrt(abs(z001-path_z[-1,0])**2 + abs(x001-path_x[-1,0])**2)
                    index = np.argmin(distance)
                    
                    
                    if distance[index] < Rmax_n*abs(path_z[-1,1]-j):
                        path_x = np.vstack((path_x, np.array([x001[index], j, i])))
                        path_z = np.vstack((path_z, np.array([z001[index], j, i])))
                    #path_x = np.vstack((path_x, np.array([x001[index], j, i])))
                    #path_z = np.vstack((path_z, np.array([z001[index], j, i])))
                    
                    nearest += 1
                else:
                    R = 1
                    vz = (path_z[-1-n_velocity, 0] - path_z[-1, 0])/(path_z[-1-n_velocity, 1] - path_z[-1, 1]) #averaged over n-1 frames
                    vx = (path_x[-1-n_velocity, 0] - path_x[-1, 0])/(path_x[-1-n_velocity, 1] - path_x[-1, 1]) #averaged over n-1 frames
                    
                    #calculate search neighbourhood
                    neighbourhood_z = path_z[-1,0] + vz
                    neighbourhood_x = path_x[-1,0] + vx
                    
                    # find points on next frame within neighbourhood
                    while True:
                        mask0 = (x[:,1] == j) &  (x[:,0] < (neighbourhood_x + R)) & (x[:,0] > (neighbourhood_x - R)) & (z[:,0] < (neighbourhood_z + R)) & (z[:,0] > (neighbourhood_z - R))
                        x001 = x[:,0][mask0]
                        z001 = z[:,0][mask0]
                        
                        if R > Rmax_n: #use nearest neighbour instead
                            # find points on next frame
                            mask0 = (x[:,1] == j)
                            x001 = x[:,0][mask0]
                            z001 = z[:,0][mask0]
                    
                            #first find closest point to previous position on camera:
                            distance = np.sqrt(abs(z001-path_z[-1,0])**2 + abs(x001-path_x[-1,0])**2)
                            index = np.argmin(distance)
                            
                            
                            if distance[index] < Rmax_v*abs(path_z[-1,1]-j):
                                path_x = np.vstack((path_x, np.array([x001[index], j, i])))
                                path_z = np.vstack((path_z, np.array([z001[index], j, i])))
                            #path_x = np.vstack((path_x, np.array([x001[index], j, i])))
                            #path_z = np.vstack((path_z, np.array([z001[index], j, i])))
                            
                            break
                        
                        if len(x001) >= 1:
                            #first find closest point to neighbourhood
                            distance = np.sqrt(abs(z001-neighbourhood_z)**2 + abs(x001-neighbourhood_x)**2)
                            index = np.argmin(distance)
                            
                            path_x = np.vstack((path_x, np.array([x001[index], j, i])))
                            path_z = np.vstack((path_z, np.array([z001[index], j, i])))
                            break
                        R += 1
    return path_x[1:,:], path_z[1:,:]



def PTV_2DMiddle(x, z, middle_frame, n_velocity=4, Rmax_v=3, Rmax_n=3, nN=200):
    # starting points, take points on the last frame, where particles are nicely separated:
    mask0 = (x[:,1] == middle_frame)
    x00 = x[:,0][mask0]
    z00 = z[:,0][mask0]
    
    path_x_back = np.zeros((1,3))
    path_z_back = np.zeros((1,3))
    
    #first, go backwards (aka, reverse)
    for i in (range(len(x00))):
        #save first coordinate
        path_x_back = np.vstack((path_x_back, np.array([x00[i], middle_frame, i])))
        path_z_back = np.vstack((path_z_back, np.array([z00[i], middle_frame, i])))

        nearest = 0
        for j in reversed(range(1, middle_frame)):
            
            # use nearest neighbour for first couple of points
            if nearest <= n_velocity+1:
                # find points on next frame
                mask0 = (x[:,1] == j)
                x001 = x[:,0][mask0]
                z001 = z[:,0][mask0]
        
                #first find closest point to previous position on camera:
                distance = np.sqrt(abs(z001-path_z_back[-1,0])**2 + abs(x001-path_x_back[-1,0])**2)
                index = np.argmin(distance)
                
                if distance[index] < Rmax_n*abs(path_z_back[-1,1]-j):
                    path_x_back = np.vstack((path_x_back, np.array([x001[index], j, i])))
                    path_z_back = np.vstack((path_z_back, np.array([z001[index], j, i])))
                
                #path_x = np.vstack((path_x, np.array([x001[index], j, i])))
                #path_z = np.vstack((path_z, np.array([z001[index], j, i])))
                
                nearest += 1
            else:
                R = 1
                vz = (path_z_back[-1-n_velocity, 0] - path_z_back[-1, 0])/(path_z_back[-1-n_velocity, 1] - path_z_back[-1, 1]) #averaged over n-1 frames
                vx = (path_x_back[-1-n_velocity, 0] - path_x_back[-1, 0])/(path_x_back[-1-n_velocity, 1] - path_x_back[-1, 1]) #averaged over n-1 frames
                
                #calculate search neighbourhood
                neighbourhood_z = path_z_back[-1,0] + vz
                neighbourhood_x = path_x_back[-1,0] + vx
                
                # find points on next frame within neighbourhood
                while True:
                    mask0 = (x[:,1] == j) &  (x[:,0] < (neighbourhood_x + R)) & (x[:,0] > (neighbourhood_x - R)) & (z[:,0] < (neighbourhood_z + R)) & (z[:,0] > (neighbourhood_z - R))
                    x001 = x[:,0][mask0]
                    z001 = z[:,0][mask0]
                    
                    if R > Rmax_n: #use nearest neighbour instead
                        # find points on next frame
                        mask0 = (x[:,1] == j)
                        x001 = x[:,0][mask0]
                        z001 = z[:,0][mask0]
                
                        #first find closest point to previous position on camera:
                        distance = np.sqrt(abs(z001-path_z_back[-1,0])**2 + abs(x001-path_x_back[-1,0])**2)
                        index = np.argmin(distance)
                        
                        #this works for some, not others >:[
                        if distance[index] < Rmax_v*abs(path_z_back[-1,1]-j):
                            path_x_back = np.vstack((path_x_back, np.array([x001[index], j, i])))
                            path_z_back = np.vstack((path_z_back, np.array([z001[index], j, i])))
                        
                        break
                    
                    if len(x001) >= 1:
                        #first find closest point to previous position on camera A:
                        distance = np.sqrt(abs(z001-neighbourhood_z)**2 + abs(x001-neighbourhood_x)**2)
                        index = np.argmin(distance)
                        
                        #if distance[index] < 4*abs(path_z[-1,1]-j):
                        
                        path_x_back = np.vstack((path_x_back, np.array([x001[index], j, i])))
                        path_z_back = np.vstack((path_z_back, np.array([z001[index], j, i])))
                        break
                    R += 1
    path_x_forw = np.zeros((1,3))
    path_z_forw = np.zeros((1,3))
    
    #second, go forwards
    for i in (range(len(x00))):
        #save first coordinate
        path_x_forw = np.vstack((path_x_forw, np.array([x00[i], middle_frame, i])))
        path_z_forw = np.vstack((path_z_forw, np.array([z00[i], middle_frame, i])))

        nearest = 0
        for j in (range(middle_frame, nN+1)):
            
            # use nearest neighbour for first couple of points
            if nearest <= n_velocity+1:
                # find points on next frame
                mask0 = (x[:,1] == j)
                x001 = x[:,0][mask0]
                z001 = z[:,0][mask0]
        
                #first find closest point to previous position on camera:
                distance = np.sqrt(abs(z001-path_z_forw[-1,0])**2 + abs(x001-path_x_forw[-1,0])**2)
                index = np.argmin(distance)
                
                if distance[index] < Rmax_n*abs(path_z_forw[-1,1]-j):
                    path_x_forw = np.vstack((path_x_forw, np.array([x001[index], j, i])))
                    path_z_forw = np.vstack((path_z_forw, np.array([z001[index], j, i])))
                
                #path_x = np.vstack((path_x, np.array([x001[index], j, i])))
                #path_z = np.vstack((path_z, np.array([z001[index], j, i])))
                
                nearest += 1
            else:
                R = 1
                vz = (path_z_forw[-1-n_velocity, 0] - path_z_forw[-1, 0])/(path_z_forw[-1-n_velocity, 1] - path_z_forw[-1, 1]) #averaged over n-1 frames
                vx = (path_x_forw[-1-n_velocity, 0] - path_x_forw[-1, 0])/(path_x_forw[-1-n_velocity, 1] - path_x_forw[-1, 1]) #averaged over n-1 frames
                
                #calculate search neighbourhood
                neighbourhood_z = path_z_forw[-1,0] + vz
                neighbourhood_x = path_x_forw[-1,0] + vx
                
                # find points on next frame within neighbourhood
                while True:
                    mask0 = (x[:,1] == j) &  (x[:,0] < (neighbourhood_x + R)) & (x[:,0] > (neighbourhood_x - R)) & (z[:,0] < (neighbourhood_z + R)) & (z[:,0] > (neighbourhood_z - R))
                    x001 = x[:,0][mask0]
                    z001 = z[:,0][mask0]
                    
                    if R > Rmax_n: #use nearest neighbour instead
                        # find points on next frame
                        mask0 = (x[:,1] == j)
                        x001 = x[:,0][mask0]
                        z001 = z[:,0][mask0]
                
                        #first find closest point to previous position on camera:
                        distance = np.sqrt(abs(z001-path_z_forw[-1,0])**2 + abs(x001-path_x_forw[-1,0])**2)
                        index = np.argmin(distance)
                        
                        #this works for some, not others >:[
                        if distance[index] < Rmax_v*abs(path_z_forw[-1,1]-j):
                            path_x_forw = np.vstack((path_x_forw, np.array([x001[index], j, i])))
                            path_z_forw = np.vstack((path_z_forw, np.array([z001[index], j, i])))
                        
                        break
                    
                    if len(x001) >= 1:
                        #first find closest point to previous position on camera A:
                        distance = np.sqrt(abs(z001-neighbourhood_z)**2 + abs(x001-neighbourhood_x)**2)
                        index = np.argmin(distance)
                        
                        #if distance[index] < 4*abs(path_z[-1,1]-j):
                        
                        path_x_forw = np.vstack((path_x_forw, np.array([x001[index], j, i])))
                        path_z_forw = np.vstack((path_z_forw, np.array([z001[index], j, i])))
                        break
                    R += 1
    #merge the two arrays
    #particle numbers should be the same
    
    path_x = np.zeros((1,3))
    path_z = np.zeros((1,3))
    
    for i in range(len(x00)):
        for j in range(1, nN+1):
            mask_forw = (path_x_forw[:,2] == i) & (path_x_forw[:,1] == j)
            p1x = path_x_forw[mask_forw]
            p1z = path_z_forw[mask_forw]
            mask_back = (path_x_back[:,2] == i) & (path_x_back[:,1] == j)
            p2x = path_x_back[mask_back]
            p2z = path_z_back[mask_back]
            
            path_x = np.vstack((path_x, p2x))
            path_x = np.vstack((path_x, p1x))
            
            path_z = np.vstack((path_z, p2z))
            path_z = np.vstack((path_z, p1z))
    
    return path_x[1:,:], path_z[1:,:]

###############################################################################
from scipy import signal

def PTV_2DMiddle_smoothing(x, z, middle_frame, n_velocity=4, Rmax_v=3, Rmax_n=3, nN=200):
    # starting points, take points on the last frame, where particles are nicely separated:
    mask0 = (x[:,1] == middle_frame)
    x00 = x[:,0][mask0]
    z00 = z[:,0][mask0]
    
    path_x_back = np.zeros((1,3))
    path_z_back = np.zeros((1,3))
    
    #first, go backwards (aka, reverse)
    for i in (range(len(x00))):
        #save first coordinate
        path_x_back = np.vstack((path_x_back, np.array([x00[i], middle_frame, i])))
        path_z_back = np.vstack((path_z_back, np.array([z00[i], middle_frame, i])))

        nearest = 0
        for j in reversed(range(1, middle_frame)):
            
            # use nearest neighbour for first couple of points
            if nearest <= n_velocity+1:
                # find points on next frame
                mask0 = (x[:,1] == j)
                x001 = x[:,0][mask0]
                z001 = z[:,0][mask0]
        
                #first find closest point to previous position on camera:
                distance = np.sqrt(abs(z001-path_z_back[-1,0])**2 + abs(x001-path_x_back[-1,0])**2)
                index = np.argmin(distance)
                
                if distance[index] < Rmax_n*abs(path_z_back[-1,1]-j):
                    path_x_back = np.vstack((path_x_back, np.array([x001[index], j, i])))
                    path_z_back = np.vstack((path_z_back, np.array([z001[index], j, i])))
                
                #path_x = np.vstack((path_x, np.array([x001[index], j, i])))
                #path_z = np.vstack((path_z, np.array([z001[index], j, i])))
                
                nearest += 1
            else:
                R = 1
                #smooth before velocity calculation
                mask_s = path_z_back[:,2] == i
                z_smooth = signal.savgol_filter(path_z_back[:,0][mask_s], n_velocity, n_velocity-1)
                x_smooth = signal.savgol_filter(path_x_back[:,0][mask_s], n_velocity, n_velocity-1)
                
                vz = (z_smooth[-1-1] - z_smooth[-1])/(path_z_back[-1-n_velocity, 1] - path_z_back[-1, 1]) #averaged over n-1 frames
                vx = (x_smooth[-1-1] - x_smooth[-1])/(path_x_back[-1-n_velocity, 1] - path_x_back[-1, 1]) #averaged over n-1 frames
                
                #calculate search neighbourhood
                neighbourhood_z = path_z_back[-1,0] + vz
                neighbourhood_x = path_x_back[-1,0] + vx
                
                # find points on next frame within neighbourhood
                while True:
                    mask0 = (x[:,1] == j) &  (x[:,0] < (neighbourhood_x + R)) & (x[:,0] > (neighbourhood_x - R)) & (z[:,0] < (neighbourhood_z + R)) & (z[:,0] > (neighbourhood_z - R))
                    x001 = x[:,0][mask0]
                    z001 = z[:,0][mask0]
                    
                    if R > Rmax_n: #use nearest neighbour instead
                        # find points on next frame
                        mask0 = (x[:,1] == j)
                        x001 = x[:,0][mask0]
                        z001 = z[:,0][mask0]
                
                        #first find closest point to previous position on camera:
                        distance = np.sqrt(abs(z001-path_z_back[-1,0])**2 + abs(x001-path_x_back[-1,0])**2)
                        index = np.argmin(distance)
                        
                        #this works for some, not others >:[
                        if distance[index] < Rmax_v*abs(path_z_back[-1,1]-j):
                            path_x_back = np.vstack((path_x_back, np.array([x001[index], j, i])))
                            path_z_back = np.vstack((path_z_back, np.array([z001[index], j, i])))
                        
                        break
                    
                    if len(x001) >= 1:
                        #first find closest point to previous position on camera A:
                        distance = np.sqrt(abs(z001-neighbourhood_z)**2 + abs(x001-neighbourhood_x)**2)
                        index = np.argmin(distance)
                        
                        #if distance[index] < 4*abs(path_z[-1,1]-j):
                        
                        path_x_back = np.vstack((path_x_back, np.array([x001[index], j, i])))
                        path_z_back = np.vstack((path_z_back, np.array([z001[index], j, i])))
                        break
                    R += 1
    path_x_forw = np.zeros((1,3))
    path_z_forw = np.zeros((1,3))
    
    #second, go forwards
    for i in (range(len(x00))):
        #save first coordinate
        path_x_forw = np.vstack((path_x_forw, np.array([x00[i], middle_frame, i])))
        path_z_forw = np.vstack((path_z_forw, np.array([z00[i], middle_frame, i])))

        nearest = 0
        for j in (range(middle_frame, nN+1)):
            
            # use nearest neighbour for first couple of points
            if nearest <= n_velocity+1:
                # find points on next frame
                mask0 = (x[:,1] == j)
                x001 = x[:,0][mask0]
                z001 = z[:,0][mask0]
        
                #first find closest point to previous position on camera:
                distance = np.sqrt(abs(z001-path_z_forw[-1,0])**2 + abs(x001-path_x_forw[-1,0])**2)
                index = np.argmin(distance)
                
                if distance[index] < Rmax_n*abs(path_z_forw[-1,1]-j):
                    path_x_forw = np.vstack((path_x_forw, np.array([x001[index], j, i])))
                    path_z_forw = np.vstack((path_z_forw, np.array([z001[index], j, i])))
                
                #path_x = np.vstack((path_x, np.array([x001[index], j, i])))
                #path_z = np.vstack((path_z, np.array([z001[index], j, i])))
                
                nearest += 1
            else:
                R = 1
                
                mask_s = path_z_back[:,2] == i
                z_smooth = signal.savgol_filter(path_z_back[:,0][mask_s], n_velocity, n_velocity-1)
                x_smooth = signal.savgol_filter(path_x_back[:,0][mask_s], n_velocity, n_velocity-1)
                
                vz = (z_smooth[-1-1] - z_smooth[-1])/(path_z_back[-1-1, 1] - path_z_back[-1, 1]) #averaged over n-1 frames
                vx = (x_smooth[-1-1] - x_smooth[-1])/(path_x_back[-1-1, 1] - path_x_back[-1, 1]) #averaged over n-1 frames
                
                #calculate search neighbourhood
                neighbourhood_z = path_z_forw[-1,0] + vz
                neighbourhood_x = path_x_forw[-1,0] + vx
                
                # find points on next frame within neighbourhood
                while True:
                    mask0 = (x[:,1] == j) &  (x[:,0] < (neighbourhood_x + R)) & (x[:,0] > (neighbourhood_x - R)) & (z[:,0] < (neighbourhood_z + R)) & (z[:,0] > (neighbourhood_z - R))
                    x001 = x[:,0][mask0]
                    z001 = z[:,0][mask0]
                    
                    if R > Rmax_n: #use nearest neighbour instead
                        # find points on next frame
                        mask0 = (x[:,1] == j)
                        x001 = x[:,0][mask0]
                        z001 = z[:,0][mask0]
                
                        #first find closest point to previous position on camera:
                        distance = np.sqrt(abs(z001-path_z_forw[-1,0])**2 + abs(x001-path_x_forw[-1,0])**2)
                        index = np.argmin(distance)
                        
                        #this works for some, not others >:[
                        if distance[index] < Rmax_v*abs(path_z_forw[-1,1]-j):
                            path_x_forw = np.vstack((path_x_forw, np.array([x001[index], j, i])))
                            path_z_forw = np.vstack((path_z_forw, np.array([z001[index], j, i])))
                        
                        break
                    
                    if len(x001) >= 1:
                        #first find closest point to previous position on camera A:
                        distance = np.sqrt(abs(z001-neighbourhood_z)**2 + abs(x001-neighbourhood_x)**2)
                        index = np.argmin(distance)
                        
                        #if distance[index] < 4*abs(path_z[-1,1]-j):
                        
                        path_x_forw = np.vstack((path_x_forw, np.array([x001[index], j, i])))
                        path_z_forw = np.vstack((path_z_forw, np.array([z001[index], j, i])))
                        break
                    R += 1
    #merge the two arrays
    #particle numbers should be the same
    
    path_x = np.zeros((1,3))
    path_z = np.zeros((1,3))
    
    for i in range(len(x00)):
        for j in range(1, nN+1):
            mask_forw = (path_x_forw[:,2] == i) & (path_x_forw[:,1] == j)
            p1x = path_x_forw[mask_forw]
            p1z = path_z_forw[mask_forw]
            mask_back = (path_x_back[:,2] == i) & (path_x_back[:,1] == j)
            p2x = path_x_back[mask_back]
            p2z = path_z_back[mask_back]
            
            path_x = np.vstack((path_x, p2x))
            path_x = np.vstack((path_x, p1x))
            
            path_z = np.vstack((path_z, p2z))
            path_z = np.vstack((path_z, p1z))
    
    return path_x[1:,:], path_z[1:,:]