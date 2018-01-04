# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:37:46 2017

@author: tanmay
"""

import os
import sys
from pdb import set_trace as keyboard
import scipy as sp
import numpy as np
import scipy.sparse as scysparse
import scipy.sparse.linalg as splinalg
import pylab as plt
import bivariate_fit as fit
import umesh_reader
import plot_data
from scipy.interpolate import griddata
from timeit import default_timer

plt.close('all')

   
def diffusion_RK2(filename,phi_cold,phi_hot,t_final,dt,alpha,flag=0):
    
    mshfile_fullpath = icemcfd_project_folder + filename
    
    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)    
    
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    faono1 = np.array(faono) # Converting faono to an array
    cold_bc = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'COLD'
    hot_bc = np.where(partofa1=='HOT')   # Vectorized approach to find face belonging to part 'HOT'
    solid = np.where(partofa1=='SOLID')   # Vectorized approach to find face belonging to part 'SOLID'
    
    # Find part to which each node belongs to
    partono = []
    for j in np.arange(nno):
        part_name = partofa1[faono1[j]]
        hot_count = np.array(np.where(part_name == 'HOT')).size
        cold_count = np.array(np.where(part_name == 'COLD')).size
        if cold_count != 0:
            partono.append('COLD')
        elif hot_count != 0:
            partono.append('HOT')
        else:
            partono.append('SOLID')
            
    partono1 = np.array(partono)
    
    phi = np.zeros(nno) #Full phi
    
    AREA = np.zeros(ncv)
    
    #Pre-processing and finding normals over all CVs
    for i in np.arange(ncv):
        nocv = noofa[faocv[i]]   # Nodal pairs for each face of the CV
        par_x = xy_no[nocv[:,1],0] - xy_no[nocv[:,0],0] #x-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        par_y = xy_no[nocv[:,1],1] - xy_no[nocv[:,0],1] #y-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        
        #Calculating areas of CV assuming rectangles or triangles
        if np.size(faocv[i]) == 3:
            area_cv = np.abs(0.5*((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])))
            AREA[i] = area_cv
            
        if np.size(faocv[i]) == 4:
            area_cv = max(np.abs((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])),np.abs((par_x[0]*par_y[2]) - (par_x[2]*par_y[0])))
            AREA[i] = area_cv
            
            
    #Temperature for internal nodes
    nno_int = nno-np.unique(noofa[cold_bc]).size-np.unique(noofa[hot_bc]).size # No. of internal nodes
    phi[:nno_int] = 400.*np.ones(nno_int) # Define initial value on internal nodes
   
    
    #Defining boundary values in phi
    phi[[np.unique(noofa[cold_bc])]] = phi_cold
    phi[[np.unique(noofa[hot_bc])]] = phi_hot
    
    Gx = scysparse.csr_matrix((nno,nno))  # Matrix to calculate X-gradient at CVs
    Gy = scysparse.csr_matrix((nno,nno))  # Matrix to calculate Y-gradient at CVs
                
    for i in np.arange(nno):
        neigh_no = np.unique(noofa[faono[i]]) # Gives neighbouring nodes including the central nodes
        neigh_no = np.delete(neigh_no,np.where(neigh_no==i)) # Find index of central node and delete that entry from neighbouring node array
        neigh_no = np.delete(neigh_no,np.where(neigh_no==-1)) # Find index of boundary node and delete the -1 entry from neighbouring node array
        dx_ik = (xy_no[neigh_no,0] - xy_no[i,0]) # Stores dx for all neighbouring nodes
        dy_ik = (xy_no[neigh_no,1] - xy_no[i,1]) # Stores dy for all neighbouring nodes
        w_ik  = 1./np.sqrt((dx_ik**2) + (dy_ik**2)) # Array of weights for least-squared fit
        a_ik = sum((w_ik*dx_ik)**2) 
        b_ik = sum(((w_ik)**2)*dx_ik*dy_ik)  #Co-efficients a_ik, b_ik, c_ik from least-squared fitting algorithm.
        c_ik = sum((w_ik*dy_ik)**2)
        
        det = (a_ik*c_ik) - (b_ik**2)       
        
        # Filling out weights for collocation point
        Gx[i,i] -= sum(((c_ik*((w_ik)**2)*dx_ik) - (b_ik*((w_ik)**2)*dy_ik)))/det
        Gy[i,i] -= sum(((a_ik*((w_ik)**2)*dy_ik) - (b_ik*((w_ik)**2)*dx_ik)))/det
        
        for j,n in enumerate(neigh_no):
            Gx[i,n] += ((c_ik*((w_ik[j])**2)*dx_ik[j]) - (b_ik*((w_ik[j])**2)*dy_ik[j]))/det
            Gy[i,n] += ((a_ik*((w_ik[j])**2)*dy_ik[j]) - (b_ik*((w_ik[j])**2)*dx_ik[j]))/det
    
    # Max. CFL based on diffusion        
    CFL_max = 4*alpha*dt/np.min(AREA)
    print "CFL_max = %f" %(CFL_max)        
    time_t = 0.
    count = 1
    Divgrad = (Gx*Gx) + (Gy*Gy)
    Divgrad[nno_int:,:] = 0.
    
    # Time loop
    while time_t <= t_final:
        
        print "Iteration %d" %(count)
        print "Time %f" %(time_t)
#        phi += dt*alpha*(Divgrad*phi)

        phi1 = phi + (dt*alpha*Divgrad*phi)
        
        phi += (dt/2.)*alpha*Divgrad*(phi + phi1)        
        
        count += 1
        time_t += dt
        if flag==1:
            if count%100==0:
                plot_data.plot_data(xy_no[:,0],xy_no[:,1],phi,"Pure_Diffusion_RK2:Solution at t=%f for CFL_max = %f.pdf" %(time_t,CFL_max))
    plot_data.plot_data(xy_no[:,0],xy_no[:,1],phi,"Pure_Diffusion_RK2:Final Solution at t=%f for CFL_max = %f.pdf" %(t_final,CFL_max))
    return (phi,CFL_max)
    
def convection_RK2(filename,phi_cold,phi_hot,t_final,dt,flag=0):
    
    mshfile_fullpath = icemcfd_project_folder + filename
    
    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)    
    
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    faono1 = np.array(faono) # Converting faono to an array
    cold_bc = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'COLD'
    hot_bc = np.where(partofa1=='HOT')   # Vectorized approach to find face belonging to part 'HOT'
    solid = np.where(partofa1=='SOLID')   # Vectorized approach to find face belonging to part 'SOLID'
    
    cx = 3.0*xy_no[:,0]
    cy = 4.0*xy_no[:,1]
    # Find part to which each node belongs to
    partono = []
    for j in np.arange(nno):
        part_name = partofa1[faono1[j]]
        hot_count = np.array(np.where(part_name == 'HOT')).size
        cold_count = np.array(np.where(part_name == 'COLD')).size
        if cold_count != 0:
            partono.append('COLD')
        elif hot_count != 0:
            partono.append('HOT')
        else:
            partono.append('SOLID')
            
    partono1 = np.array(partono)
    
    phi = np.zeros(nno) #Full phi
    
    AREA = np.zeros(ncv)
    
    #Pre-processing and finding normals over all CVs
    for i in np.arange(ncv):
        nocv = noofa[faocv[i]]   # Nodal pairs for each face of the CV
        par_x = xy_no[nocv[:,1],0] - xy_no[nocv[:,0],0] #x-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        par_y = xy_no[nocv[:,1],1] - xy_no[nocv[:,0],1] #y-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        
        #Calculating areas of CV assuming rectangles or triangles
        if np.size(faocv[i]) == 3:
            area_cv = np.abs(0.5*((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])))
            AREA[i] = area_cv
            
        if np.size(faocv[i]) == 4:
            area_cv = max(np.abs((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])),np.abs((par_x[0]*par_y[2]) - (par_x[2]*par_y[0])))
            AREA[i] = area_cv
            
            
    #Temperature for internal nodes
    nno_int = nno-np.unique(noofa[cold_bc]).size-np.unique(noofa[hot_bc]).size # No. of internal nodes
    phi[:nno_int] = 400.*np.ones(nno_int) # Define initial value on internal nodes
   
    
    #Defining boundary values in phi
    phi[[np.unique(noofa[cold_bc])]] = phi_cold
    phi[[np.unique(noofa[hot_bc])]] = phi_hot
    
    Gx = scysparse.csr_matrix((nno,nno))  # Matrix to calculate X-gradient at CVs
    Gy = scysparse.csr_matrix((nno,nno))  # Matrix to calculate Y-gradient at CVs
                
    for i in np.arange(nno):
        neigh_no = np.unique(noofa[faono[i]]) # Gives neighbouring CVs including the central CV
        neigh_no = np.delete(neigh_no,np.where(neigh_no==i)) # Find index of central CV and delete that entry from neighbouring CV array
        neigh_no = np.delete(neigh_no,np.where(neigh_no==-1)) # Find index of boundary CV and delete the -1 entry from neighbouring CV array
        dx_ik = (xy_no[neigh_no,0] - xy_no[i,0]) # Stores dx for all neighbouring CVs
        dy_ik = (xy_no[neigh_no,1] - xy_no[i,1]) # Stores dy for all neighbouring CVs
        w_ik  = 1./np.sqrt((dx_ik**2) + (dy_ik**2)) # Array of weights for least-squared fit
        a_ik = sum((w_ik*dx_ik)**2) 
        b_ik = sum(((w_ik)**2)*dx_ik*dy_ik)  #Co-efficients a_ik, b_ik, c_ik from least-squared fitting algorithm.
        c_ik = sum((w_ik*dy_ik)**2)
        
        det = (a_ik*c_ik) - (b_ik**2)       
        
        # Filling out weights for collocation point
        Gx[i,i] -= sum(((c_ik*((w_ik)**2)*dx_ik) - (b_ik*((w_ik)**2)*dy_ik)))/det
        Gy[i,i] -= sum(((a_ik*((w_ik)**2)*dy_ik) - (b_ik*((w_ik)**2)*dx_ik)))/det
        
        for j,n in enumerate(neigh_no):
            Gx[i,n] += ((c_ik*((w_ik[j])**2)*dx_ik[j]) - (b_ik*((w_ik[j])**2)*dy_ik[j]))/det
            Gy[i,n] += ((a_ik*((w_ik[j])**2)*dy_ik[j]) - (b_ik*((w_ik[j])**2)*dx_ik[j]))/det
    
    # Max. CFL based on diffusion        
    CFL_max = np.max(np.sqrt((cx)**2 + (cy)**2))*dt/np.sqrt(np.min(AREA))
    print "CFL_max = %f" %(CFL_max)        
    time_t = 0.
    count = 1
    
    # Time loop
    while time_t <= t_final:
        
        print "Iteration %d" %(count)
        print "Time %f" %(time_t)
        
        # First RK substep
        phi_x1 = Gx*phi
        phi_y1 = Gy*phi
        phi1 = (dt*(np.multiply(cx,phi_x1) + np.multiply(cy,phi_y1)))
        
        # 2nd RK substep
        phi_x2 = Gx*phi1
        phi_y2 = Gy*phi1
        
        # Final prediction
        phi_x_star = phi_x1 + phi_x2
        phi_y_star = phi_y1 + phi_y2
        phi += (dt/2.)*(np.multiply(cx,phi_x_star)+ np.multiply(cy,phi_y_star))        
        
        count += 1
        time_t += dt
        if flag==1:
            if count%100==0:
                plot_data.plot_data(xy_no[:,0],xy_no[:,1],phi,"Pure_Convection_RK2:Solution at t=%f at CFL_max = %f.pdf" %(time_t,CFL_max))
    plot_data.plot_data(xy_no[:,0],xy_no[:,1],phi,"Pure_Convection_RK2:Final Solution at t=%f at CFL_max = %f.pdf" %(t_final,CFL_max))
    return (phi,CFL_max)
    
def convection_SL(filename,phi_cold,phi_hot,t_final,dt):
    
    mshfile_fullpath = icemcfd_project_folder + filename
    
    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)    
    
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    faono1 = np.array(faono) # Converting faono to an array
    cold_bc = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'COLD'
    hot_bc = np.where(partofa1=='HOT')   # Vectorized approach to find face belonging to part 'HOT'
    solid = np.where(partofa1=='SOLID')   # Vectorized approach to find face belonging to part 'SOLID'
    
    cx = 3.0*xy_no[:,0]
    cy = 4.0*xy_no[:,1]
    # Find part to which each node belongs to
    partono = []
    for j in np.arange(nno):
        part_name = partofa1[faono1[j]]
        hot_count = np.array(np.where(part_name == 'HOT')).size
        cold_count = np.array(np.where(part_name == 'COLD')).size
        if cold_count != 0:
            partono.append('COLD')
        elif hot_count != 0:
            partono.append('HOT')
        else:
            partono.append('SOLID')
            
    partono1 = np.array(partono)
    
    phi = np.zeros(nno) #Full phi
    
    AREA = np.zeros(ncv)
    
    #Pre-processing and finding normals over all CVs
    for i in np.arange(ncv):
        nocv = noofa[faocv[i]]   # Nodal pairs for each face of the CV
        par_x = xy_no[nocv[:,1],0] - xy_no[nocv[:,0],0] #x-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        par_y = xy_no[nocv[:,1],1] - xy_no[nocv[:,0],1] #y-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        
        #Calculating areas of CV assuming rectangles or triangles
        if np.size(faocv[i]) == 3:
            area_cv = np.abs(0.5*((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])))
            AREA[i] = area_cv
            
        if np.size(faocv[i]) == 4:
            area_cv = max(np.abs((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])),np.abs((par_x[0]*par_y[2]) - (par_x[2]*par_y[0])))
            AREA[i] = area_cv
            
            
    #Temperature for internal nodes
    nno_int = nno-np.unique(noofa[cold_bc]).size-np.unique(noofa[hot_bc]).size # No. of internal nodes
    phi[:nno_int] = 400.*np.ones(nno_int) # Define initial value on internal nodes
   
    
    #Defining boundary values in phi
    phi[[np.unique(noofa[cold_bc])]] = phi_cold
    phi[[np.unique(noofa[hot_bc])]] = phi_hot
    
    # Defining the operator
    x_past = np.zeros(nno_int)
    y_past = np.zeros(nno_int)
    phi_past = np.zeros(nno_int)
    
    # Filling weights in L
    for i in np.arange(nno_int):
        
        
        x_past[i] = xy_no[i,0] - cx[i]*dt
        y_past[i] = xy_no[i,1] - cy[i]*dt
        
        
    
    # Max. CFL based on diffusion        
    CFL_max = np.max(np.sqrt((cx)**2 + (cy)**2))*dt/np.sqrt(np.min(AREA))
    print "CFL_max = %f" %(CFL_max)        
    time_t = 0.
    count = 1
    
    # Time loop
    while time_t <= t_final:
        
        print "Iteration %d" %(count)
        print "Time %f" %(time_t)
        phi_past = griddata(np.vstack((xy_no[:,0].flatten(),xy_no[:,1].flatten())).T, np.vstack(phi.flatten()),(x_past,y_past),method="cubic")
        phi[:nno_int] = phi_past.reshape(phi[:nno_int].shape)
        count += 1
        time_t += dt
#        if count%20:
#            plot_data.plot_data(xy_no[:,0],xy_no[:,1],phi,"Solution at t=%f.pdf" %(time_t))
#    plot_data.plot_data(xy_no[:,0],xy_no[:,1],phi,"Final Solution at t=%f.pdf" %(t_final))
    print "CFL_max = %f" %(CFL_max)     
    plot_data.plot_data(xy_no[:,0],xy_no[:,1],phi,"Final Solution: Convection at t=%f for Semi-Lagrange approach.pdf" %(time_t)) 
    return (phi,CFL_max)
    
def diffusion_CN(filename,phi_cold,phi_hot,t_final,dt,alpha,flag=0):
    
    mshfile_fullpath = icemcfd_project_folder + filename
    
    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)    
    
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    faono1 = np.array(faono) # Converting faono to an array
    cold_bc = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'COLD'
    hot_bc = np.where(partofa1=='HOT')   # Vectorized approach to find face belonging to part 'HOT'
    solid = np.where(partofa1=='SOLID')   # Vectorized approach to find face belonging to part 'SOLID'
    
    # Find part to which each node belongs to
    partono = []
    for j in np.arange(nno):
        part_name = partofa1[faono1[j]]
        hot_count = np.array(np.where(part_name == 'HOT')).size
        cold_count = np.array(np.where(part_name == 'COLD')).size
        if cold_count != 0:
            partono.append('COLD')
        elif hot_count != 0:
            partono.append('HOT')
        else:
            partono.append('SOLID')
            
    partono1 = np.array(partono)
    
    phi = np.zeros(nno) #Full phi
    
    AREA = np.zeros(ncv)
    
    #Pre-processing and finding normals over all CVs
    for i in np.arange(ncv):
        nocv = noofa[faocv[i]]   # Nodal pairs for each face of the CV
        par_x = xy_no[nocv[:,1],0] - xy_no[nocv[:,0],0] #x-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        par_y = xy_no[nocv[:,1],1] - xy_no[nocv[:,0],1] #y-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        
        #Calculating areas of CV assuming rectangles or triangles
        if np.size(faocv[i]) == 3:
            area_cv = np.abs(0.5*((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])))
            AREA[i] = area_cv
            
        if np.size(faocv[i]) == 4:
            area_cv = max(np.abs((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])),np.abs((par_x[0]*par_y[2]) - (par_x[2]*par_y[0])))
            AREA[i] = area_cv
            
            
    #Temperature for internal nodes
    nno_int = nno-np.unique(noofa[cold_bc]).size-np.unique(noofa[hot_bc]).size # No. of internal nodes
    phi[:nno_int] = 400.*np.ones(nno_int) # Define initial value on internal nodes
   
    
    #Defining boundary values in phi
    phi[[np.unique(noofa[cold_bc])]] = phi_cold
    phi[[np.unique(noofa[hot_bc])]] = phi_hot
    
    Gx = scysparse.csr_matrix((nno,nno))  # Matrix to calculate X-gradient at CVs
    Gy = scysparse.csr_matrix((nno,nno))  # Matrix to calculate Y-gradient at CVs
    I = scysparse.identity(nno)
                
    for i in np.arange(nno):
        neigh_no = np.unique(noofa[faono[i]]) # Gives neighbouring nodes including the central nodes
        neigh_no = np.delete(neigh_no,np.where(neigh_no==i)) # Find index of central node and delete that entry from neighbouring node array
        neigh_no = np.delete(neigh_no,np.where(neigh_no==-1)) # Find index of boundary node and delete the -1 entry from neighbouring node array
        dx_ik = (xy_no[neigh_no,0] - xy_no[i,0]) # Stores dx for all neighbouring nodes
        dy_ik = (xy_no[neigh_no,1] - xy_no[i,1]) # Stores dy for all neighbouring nodes
        w_ik  = 1./np.sqrt((dx_ik**2) + (dy_ik**2)) # Array of weights for least-squared fit
        a_ik = sum((w_ik*dx_ik)**2) 
        b_ik = sum(((w_ik)**2)*dx_ik*dy_ik)  #Co-efficients a_ik, b_ik, c_ik from least-squared fitting algorithm.
        c_ik = sum((w_ik*dy_ik)**2)
        
        det = (a_ik*c_ik) - (b_ik**2)       
        
        # Filling out weights for collocation point
        Gx[i,i] -= sum(((c_ik*((w_ik)**2)*dx_ik) - (b_ik*((w_ik)**2)*dy_ik)))/det
        Gy[i,i] -= sum(((a_ik*((w_ik)**2)*dy_ik) - (b_ik*((w_ik)**2)*dx_ik)))/det
        
        for j,n in enumerate(neigh_no):
            Gx[i,n] += ((c_ik*((w_ik[j])**2)*dx_ik[j]) - (b_ik*((w_ik[j])**2)*dy_ik[j]))/det
            Gy[i,n] += ((a_ik*((w_ik[j])**2)*dy_ik[j]) - (b_ik*((w_ik[j])**2)*dx_ik[j]))/det
            
    
    
    # Max. CFL based on diffusion        
    CFL_max = 4*alpha*dt/np.min(AREA)
    print "CFL_max = %f" %(CFL_max)        
    time_t = 0.
    count = 1
    
    Divgrad = (Gx*Gx) + (Gy*Gy)
    Divgrad[nno_int:,:] = 0.
    
    A = scysparse.identity(nno) - (0.5*dt*Divgrad)
    B = scysparse.identity(nno) + (0.5*dt*Divgrad)    
        
    # Time loop
    while time_t <= t_final:
        
        print "Iteration %d" %(count)
        print "Time %f" %(time_t)
        phi = splinalg.spsolve(A,B*phi)
        count += 1
        time_t += dt
        if flag==1:
            if count%10==0:
                plot_data.plot_data(xy_no[:,0],xy_no[:,1],phi,"Pure Diffusion: Crank Nicolson, Solution at t=%f, CFL = %f.pdf" %(time_t,CFL_max))
    plot_data.plot_data(xy_no[:,0],xy_no[:,1],phi,"Pure Diffusion: Crank Nicolson, Final Solution at t=%f at CFL_max = %f.pdf" %(t_final, CFL_max))
    return (phi,CFL_max)
    
p1a = True # CFL plots based on advection and diffusion time scales.
p1b = True  # Pure diffusion using above value of alpha 
p1c = True
p1SL = True
p1d = True
psp = True

figure_folder = "../report/figures/"
icemcfd_project_folder = './mesh/'
filename = "Mesh_3.msh"
filename1 = "Mesh_2.msh"

figwidth,figheight = 14,12
lineWidth = 3
fontSize = 25
gcafontSize = 21

if p1a:
    
    print "Module for problem 1a."
    mshfile_fullpath = icemcfd_project_folder + filename
    
    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)
    
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    faono1 = np.array(faono) # Converting faono to an array
#    cold_bc = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'COLD'
#    hot_bc = np.where(partofa1=='HOT')   # Vectorized approach to find face belonging to part 'HOT'
#    solid = np.where(partofa1=='SOLID')   # Vectorized approach to find face belonging to part 'SOLID'
    
    CFL_max = 0.85

    cx = 3.0*xy_no[:,0]
    cy = 4.0*xy_no[:,1]
    
    AREA = np.zeros(ncv)
    C_avg = np.zeros(ncv)
    
    #Pre-processing and finding areas of all CVs
    for i in np.arange(ncv):
        nocv = noofa[faocv[i]]   # Nodal pairs for each face of the CV  
        C_avg[i] = np.average(np.sqrt((cx[np.unique(nocv)]**2) + (cy[np.unique(nocv)]**2))) # Finds average |c| at CV        
        par_x = xy_no[nocv[:,1],0] - xy_no[nocv[:,0],0] #x-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        par_y = xy_no[nocv[:,1],1] - xy_no[nocv[:,0],1] #y-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        
        #Calculating areas of CV assuming rectangles or triangles
        if np.size(faocv[i]) == 3:
            area_cv = np.abs(0.5*((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])))
            AREA[i] = area_cv
            
        if np.size(faocv[i]) == 4:
            area_cv = max(np.abs((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])),np.abs((par_x[0]*par_y[2]) - (par_x[2]*par_y[0])))
            AREA[i] = area_cv
            
    dt_conv = CFL_max*np.sqrt(AREA)/C_avg # delta_t_conv considering CFL_max of 0.85 for pure convection
    dt_diff_min = np.min(dt_conv)
    
    print "The minimum time step is %f" %(dt_diff_min)
    
    alpha = CFL_max*AREA/(4.*dt_diff_min) # delta_t_conv considering CFL_max of 0.85 for pure diffusion
    
    ALPHA = np.min(alpha)
    print r"$\alpha$ = %f" %(ALPHA)
    
    CFL = np.zeros(ncv)
    
    
    for i in np.arange(ncv):
        CFL[i] = max(4.*ALPHA*dt_diff_min/(AREA[i]),C_avg[i]*dt_diff_min/np.sqrt(AREA[i]))
        
    plot_data.plot_data(xy_cv[:,0],xy_cv[:,1],CFL,"CFL_plot.pdf")
    print ".............................................................................."
    
    
if p1b:
    
    print "Module for problem 1b."
    
    mshfile_fullpath = icemcfd_project_folder + filename
    
    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)
    
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    faono1 = np.array(faono) # Converting faono to an array
#    cold_bc = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'COLD'
#    hot_bc = np.where(partofa1=='HOT')   # Vectorized approach to find face belonging to part 'HOT'
#    solid = np.where(partofa1=='SOLID')   # Vectorized approach to find face belonging to part 'SOLID'
    
    CFL_max = 0.85

    cx = 3.0*xy_no[:,0]
    cy = 4.0*xy_no[:,1]
    
    AREA = np.zeros(ncv)
    C_avg = np.zeros(ncv)
    
    #Pre-processing and finding areas of all CVs
    for i in np.arange(ncv):
        nocv = noofa[faocv[i]]   # Nodal pairs for each face of the CV  
        C_avg[i] = np.average(np.sqrt((cx[np.unique(nocv)]**2) + (cy[np.unique(nocv)]**2))) # Finds average |c| at CV        
        par_x = xy_no[nocv[:,1],0] - xy_no[nocv[:,0],0] #x-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        par_y = xy_no[nocv[:,1],1] - xy_no[nocv[:,0],1] #y-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        
        #Calculating areas of CV assuming rectangles or triangles
        if np.size(faocv[i]) == 3:
            area_cv = np.abs(0.5*((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])))
            AREA[i] = area_cv
            
        if np.size(faocv[i]) == 4:
            area_cv = max(np.abs((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])),np.abs((par_x[0]*par_y[2]) - (par_x[2]*par_y[0])))
            AREA[i] = area_cv
            
    dt_conv = CFL_max*np.sqrt(AREA)/C_avg # delta_t_conv considering CFL_max of 0.85 for pure convection
    dt_diff_min = np.min(dt_conv)
    
    print "The minimum time step is %f" %(dt_diff_min)
    
    alpha = CFL_max*AREA/(4.*dt_diff_min) # delta_t_conv considering CFL_max of 0.85 for pure diffusion
    
    ALPHA = np.min(alpha)
    
    CFL = np.zeros(ncv)
    
    
    for i in np.arange(ncv):
        CFL[i] = max(4.*ALPHA*dt_diff_min/(AREA[i]),C_avg[i]*dt_diff_min/np.sqrt(AREA[i]))
        
    phi,CFL_max = diffusion_RK2(filename,300,500,1.,0.001,ALPHA,1)
    phi,CFL_max = diffusion_RK2(filename,300,500,10.,0.001,ALPHA,0)
    phi,CFL_max3 = diffusion_RK2(filename,300,500,1.,0.039,ALPHA,0)
    print ".............................................................................."
    
if p1c:
    print "Module for problem 1c."
    phi,CFL_max = convection_RK2(filename,300,500,1.,0.001,1)
    phi,CFL_max2 = convection_RK2(filename,300,500,1.,0.00002,0)
    print ".............................................................................."
    
if p1SL:
    print "Module for Semi-Lagrange approach."
    phi,CFL_max = convection_SL(filename,300,500,1,0.005)
    print ".............................................................................."    
    
if p1d:
    print "Module for problem 1d."
    mshfile_fullpath = icemcfd_project_folder + filename
    
    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)
    
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    faono1 = np.array(faono) # Converting faono to an array
#    cold_bc = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'COLD'
#    hot_bc = np.where(partofa1=='HOT')   # Vectorized approach to find face belonging to part 'HOT'
#    solid = np.where(partofa1=='SOLID')   # Vectorized approach to find face belonging to part 'SOLID'
    
    CFL_max = 0.85

    cx = 3.0*xy_no[:,0]
    cy = 4.0*xy_no[:,1]
    
    AREA = np.zeros(ncv)
    C_avg = np.zeros(ncv)
    
    #Pre-processing and finding areas of all CVs
    for i in np.arange(ncv):
        nocv = noofa[faocv[i]]   # Nodal pairs for each face of the CV  
        C_avg[i] = np.average(np.sqrt((cx[np.unique(nocv)]**2) + (cy[np.unique(nocv)]**2))) # Finds average |c| at CV        
        par_x = xy_no[nocv[:,1],0] - xy_no[nocv[:,0],0] #x-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        par_y = xy_no[nocv[:,1],1] - xy_no[nocv[:,0],1] #y-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        
        #Calculating areas of CV assuming rectangles or triangles
        if np.size(faocv[i]) == 3:
            area_cv = np.abs(0.5*((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])))
            AREA[i] = area_cv
            
        if np.size(faocv[i]) == 4:
            area_cv = max(np.abs((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])),np.abs((par_x[0]*par_y[2]) - (par_x[2]*par_y[0])))
            AREA[i] = area_cv
            
    dt_conv = CFL_max*np.sqrt(AREA)/C_avg # delta_t_conv considering CFL_max of 0.85 for pure convection
    dt_diff_min = np.min(dt_conv)
    
    print "The minimum time step is %f" %(dt_diff_min)
    
    alpha = CFL_max*AREA/(4.*dt_diff_min) # delta_t_conv considering CFL_max of 0.85 for pure diffusion
    
    ALPHA = np.min(alpha)
    
    CFL = np.zeros(ncv)
    
    
    for i in np.arange(ncv):
        CFL[i] = max(4.*ALPHA*dt_diff_min/(AREA[i]),C_avg[i]*dt_diff_min/np.sqrt(AREA[i]))
        

    phi,CFL_max = diffusion_CN(filename1,300,500,1.,0.01,ALPHA,1)
    phi,CFL_max = diffusion_CN(filename1,300,500,1.,0.1,ALPHA)
    print ".............................................................................."
    
if psp:
    print "Problem 2"
    N = 10
    n = np.arange(0,N+1)
    
    xj = 0.5*(1 - np.cos(n*np.pi/N))
    yj = 0.5*(1 - np.cos(n*np.pi/N))    
    
    X_flux,Y_flux = np.meshgrid(xj,yj)
    plt.plot(X_flux,Y_flux,'k')
    plt.plot(Y_flux,X_flux,'k')
    
    xj = 0.5*(1 - np.cos(((2*n[:-1]) + 1)*np.pi/(2*N)))
#    xj = np.concatenate([[0.],xj,[1.]])
    yj = 0.5*(1 - np.cos(((2*n[:-1]) + 1)*np.pi/(2*N)))
#    yj = np.concatenate([[0.],yj,[1.]])

    X_sol,Y_sol = np.meshgrid(xj,yj)
    plt.plot(X_sol,Y_sol,'r--')
    plt.plot(Y_sol,X_sol,'r--')
    
    figure_name = figure_folder + "Solution and flux points.pdf"
    print "Saving figure: "+ figure_name
    plt.savefig(figure_name)
    plt.close()
    print ".............................................................................."     
    
    # Interpolation from solution to flux points
    L_s_f = np.zeros((X_sol[0,:].size,X_flux[0,:].size))
    Dx_f_s = np.zeros((X_flux[0,:].size,X_sol[0,:].size))
    derivative_order = 1
    
    for i,x in enumerate(X_sol[0,:]):
        
        x_stencil = X_sol[0,:]
        N = x_stencil.size
        if x_stencil.ndim>1:
            sys.exit("stencil array is not a 1D numpy array")
        
        polynomial_order = len(x_stencil)-1
        base_func = np.zeros(N,)
        base_func[i] = 1.0
        
        poly_coefs = np.polyfit(x_stencil,base_func,polynomial_order)
        
        for j,y in enumerate(X_flux[0,:]):
            
            L_s_f[i,j] = np.polyval(poly_coefs,y)
            
    # Evaluate 1st derivative        
    for i,x in enumerate(X_flux[0,:]):
        
        x_stencil = X_flux[0,:]
        N = x_stencil.size
        if x_stencil.ndim>1:
            sys.exit("stencil array is not a 1D numpy array")
        
        polynomial_order = len(x_stencil)-1
        base_func = np.zeros(N,)
        base_func[i] = 1.0
        
        poly_coefs = np.polyder(np.polyfit(x_stencil,base_func,polynomial_order),derivative_order)
        
        for j,y in enumerate(X_sol[0,:]):
            
            Dx_f_s[i,j] = np.polyval(poly_coefs,y)
            
    print" Block for verification of 1st derivative evaluation"
    print "f(x) = x, f(y) = y"
    # Validation of 1st x and y derivative
    phi = X_sol
    
    phi_flux = phi.dot(L_s_f)
    phi_flux[:,0] = X_flux[:-1,0]
    phi_flux[:,-1] = X_flux[:-1,-1]
    
    phi_der = phi_flux.dot(Dx_f_s)
    print "The first derivative of f(x) is:"
    print phi_der
    
    phi = Y_sol
    
    phi_flux = np.transpose(L_s_f).dot(phi)
    phi_flux[0,:] = Y_flux[0,:-1]
    phi_flux[-1,:] = Y_flux[-1,:-1]
    
    phi_der = np.transpose(Dx_f_s).dot(phi_flux)
    print "The first derivative of f(y) is:"
    print phi_der
    
    print ".............................................................................."
    print" Block for verification of 2nd derivative evaluation"
    print "f(x) = x^2, f(y) = y^2"
    # Validation of 2nd x and y derivative
    phi = X_sol*X_sol
    
    phi_flux = phi.dot(L_s_f)
    phi_flux[:,0] = X_flux[:-1,0]*X_flux[:-1,0]
    phi_flux[:,-1] = X_flux[:-1,-1]*X_flux[:-1,-1]
    phi_der = phi_flux.dot(Dx_f_s)
    
    phi_flux_flux = phi_der.dot(L_s_f)
    phi_der_der = phi_flux_flux.dot(Dx_f_s)     
    print "The 2nd derivative of f(x) is:"
    print phi_der_der
    
    phi = Y_sol*Y_sol
    
    phi_flux = np.transpose(L_s_f).dot(phi)
    phi_flux[0,:] = Y_flux[0,:-1]*Y_flux[0,:-1]
    phi_flux[-1,:] = Y_flux[-1,:-1]*Y_flux[-1,:-1]
    phi_der = np.transpose(Dx_f_s).dot(phi_flux)
    
    phi_flux_flux = np.transpose(L_s_f).dot(phi_der)
    phi_der_der = np.transpose(Dx_f_s).dot(phi_flux_flux)
    print "The 2nd derivative of f(y) is:"
    print phi_der_der
    
    print ".............................................................................."
    # Explicit time advancement diffusion
    print" Pure diffusion"
    phi = np.zeros(X_sol.shape)    
    phi_bc = 500.
    
    alpha=0.01
    dt =0.001
    
    final_t = 100.
    t = 0.
    it = 1    
    
    while t<final_t:
        
        phi_flux_x = phi.dot(L_s_f)
        phi_flux_x[:,0] = phi_bc
        phi_flux_x[:,-1] = phi_bc
        phi_der_x = phi_flux_x.dot(Dx_f_s)
        phi_flux_flux_x = phi_der_x.dot(L_s_f)
        phi_der_der_x = phi_flux_flux_x.dot(Dx_f_s)  
        
        phi_flux_y = np.transpose(L_s_f).dot(phi)
        phi_flux_y[0,:] = phi_bc
        phi_flux_y[-1,:] = phi_bc
        phi_der_y = np.transpose(Dx_f_s).dot(phi_flux_y)
        phi_flux_flux_y = np.transpose(L_s_f).dot(phi_der_y)
        phi_der_der_y = np.transpose(Dx_f_s).dot(phi_flux_flux_y)
        
        phi1 = phi + (alpha*(dt/2.)*(phi_der_der_x + phi_der_der_y))
        
        
        phi_flux_x1 = phi1.dot(L_s_f)
        phi_flux_x1[:,0] = phi_bc
        phi_flux_x1[:,-1] = phi_bc
        phi_der_x1 = phi_flux_x1.dot(Dx_f_s)
        phi_flux_flux_x1 = phi_der_x1.dot(L_s_f)
        phi_der_der_x1 = phi_flux_flux_x1.dot(Dx_f_s)  
        
        phi_flux_y1 = np.transpose(L_s_f).dot(phi1)
        phi_flux_y1[0,:] = phi_bc
        phi_flux_y1[-1,:] = phi_bc
        phi_der_y1 = np.transpose(Dx_f_s).dot(phi_flux_y1)
        phi_flux_flux_y1 = np.transpose(L_s_f).dot(phi_der_y1)
        phi_der_der_y1 = np.transpose(Dx_f_s).dot(phi_flux_flux_y1)
        
        phi2 = phi + (alpha*(dt/2.)*(phi_der_der_x1 + phi_der_der_y1))
        
        
        phi_flux_x2 = phi2.dot(L_s_f)
        phi_flux_x2[:,0] = phi_bc
        phi_flux_x2[:,-1] = phi_bc
        phi_der_x2 = phi_flux_x2.dot(Dx_f_s)
        phi_flux_flux_x2 = phi_der_x2.dot(L_s_f)
        phi_der_der_x2 = phi_flux_flux_x2.dot(Dx_f_s)  
        
        phi_flux_y2 = np.transpose(L_s_f).dot(phi2)
        phi_flux_y2[0,:] = phi_bc
        phi_flux_y2[-1,:] = phi_bc
        phi_der_y2 = np.transpose(Dx_f_s).dot(phi_flux_y2)
        phi_flux_flux_y2 = np.transpose(L_s_f).dot(phi_der_y2)
        phi_der_der_y2 = np.transpose(Dx_f_s).dot(phi_flux_flux_y2)
        
        phi3 = phi + (alpha*dt*(phi_der_der_x2 + phi_der_der_y2))
        
        
        phi_flux_x3 = phi3.dot(L_s_f)
        phi_flux_x3[:,0] = phi_bc
        phi_flux_x3[:,-1] = phi_bc
        phi_der_x3 = phi_flux_x3.dot(Dx_f_s)
        phi_flux_flux_x3 = phi_der_x3.dot(L_s_f)
        phi_der_der_x3 = phi_flux_flux_x3.dot(Dx_f_s)  
        
        phi_flux_y3 = np.transpose(L_s_f).dot(phi3)
        phi_flux_y3[0,:] = phi_bc
        phi_flux_y3[-1,:] = phi_bc
        phi_der_y3 = np.transpose(Dx_f_s).dot(phi_flux_y3)
        phi_flux_flux_y3 = np.transpose(L_s_f).dot(phi_der_y3)
        phi_der_der_y3 = np.transpose(Dx_f_s).dot(phi_flux_flux_y3)
        
        phi += alpha*(dt/6.)*((phi_der_der_x + phi_der_der_y) + (2*(phi_der_der_x1 + phi_der_der_y1)) + (2*(phi_der_der_x2 + phi_der_der_y2)) + (phi_der_der_x3 + phi_der_der_y3))
        
        
        t += dt
        print "Iteration %d" %(it)
        it +=1
        
        
    plt.contourf(X_sol,Y_sol,phi)   
    plt.colorbar()
    figure_name = figure_folder + "Pure diffusion solved on a unit domain.pdf"
    print "Saving figure: "+ figure_name
    plt.savefig(figure_name)
    plt.close()  
    print ".............................................................................."
    # Explicit time advancement advection
    print" Pure advection"
    phi = np.zeros(X_sol.shape)    
    phi_bc = 500.
    
    cx = 3.0*np.ones(X_sol.shape)
    cy = 3.0*np.ones(X_sol.shape)
    dt =0.0001
    
    final_t = 1.
    t = 0.
    it = 1    
    
    while t<final_t:
        
        phi_flux_x = phi.dot(L_s_f)
        phi_flux_x[:,0] = phi_bc
#        phi_flux_x[:,-1] = phi_bc
        phi_der_x = phi_flux_x.dot(Dx_f_s)
        
        phi_flux_y = np.transpose(L_s_f).dot(phi)
        phi_flux_y[0,:] = phi_bc
#        phi_flux_y[-1,:] = phi_bc
        phi_der_y = np.transpose(Dx_f_s).dot(phi_flux_y)
        
        phi1 = phi - ((dt/2.)*((cx*phi_der_x) + ((cy*phi_der_y))))
        
        
        phi_flux_x1 = phi1.dot(L_s_f)
        phi_flux_x1[:,0] = phi_bc
#        phi_flux_x[:,-1] = phi_bc
        phi_der_x1 = phi_flux_x1.dot(Dx_f_s)
        
        phi_flux_y1 = np.transpose(L_s_f).dot(phi1)
        phi_flux_y1[0,:] = phi_bc
#        phi_flux_y[-1,:] = phi_bc
        phi_der_y1 = np.transpose(Dx_f_s).dot(phi_flux_y1)
        
        phi2 = phi - ((dt/2.)*((cx*phi_der_x1) + ((cy*phi_der_y1))))
        
        
        phi_flux_x2 = phi2.dot(L_s_f)
        phi_flux_x2[:,0] = phi_bc
#        phi_flux_x[:,-1] = phi_bc
        phi_der_x2 = phi_flux_x2.dot(Dx_f_s)
        
        phi_flux_y2 = np.transpose(L_s_f).dot(phi2)
        phi_flux_y2[0,:] = phi_bc
#        phi_flux_y[-1,:] = phi_bc
        phi_der_y2 = np.transpose(Dx_f_s).dot(phi_flux_y2)
        
        phi3 = phi - (dt*((cx*phi_der_x2) + ((cy*phi_der_y2))))
        
        
        phi_flux_x3 = phi3.dot(L_s_f)
        phi_flux_x3[:,0] = phi_bc
#        phi_flux_x[:,-1] = phi_bc
        phi_der_x3 = phi_flux_x3.dot(Dx_f_s)
        
        phi_flux_y3 = np.transpose(L_s_f).dot(phi3)
        phi_flux_y3[0,:] = phi_bc
#        phi_flux_y[-1,:] = phi_bc
        phi_der_y3 = np.transpose(Dx_f_s).dot(phi_flux_y3)
        
        phi = phi - ((dt/6.)*(((cx*phi_der_x) + (cy*phi_der_y)) + (2*((cx*phi_der_x1) + (cy*phi_der_y1))) + (2*((cx*phi_der_x2) + (cy*phi_der_y2))) + ((cx*phi_der_x3) + (cy*phi_der_y3))))
        
        print np.average(phi)
        t += dt
        print "Iteration %d" %(it)
        it +=1
#        keyboard()
        
        
    plt.contourf(X_sol,Y_sol,phi)   
    plt.colorbar()
    figure_name = figure_folder + "Pure advection solved on a unit domain.pdf"
    print "Saving figure: "+ figure_name
    plt.savefig(figure_name)
    plt.close()  
        
    
    print ".............................................................................."
    
    
