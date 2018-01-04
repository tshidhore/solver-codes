# -*- coding: utf-8 -*-
"""

Module for evaluating the gradient

source                                    destination               function

nodes                                     nodes                     Gradient_no
CVs                                       CVs                       Gradient_CV
faces                                     faces                     Gradient_fa

Technique: Least-squared fitting

Created on Fri Nov 17 11:29:24 2017

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
#import bivariate_fit as fit
import umesh_reader
#import plot_data
from scipy.interpolate import griddata
from timeit import default_timer


def Gradient_cvfa(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa):
    
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]
    
    partofa1 = np.array(partofa) # Converting partofa to an array
    faono1 = np.array(faono) # Converting faono to an array
    
    Gx_int = scysparse.csr_matrix((ncv,ncv))  # Matrix to calculate X-gradient at CVs
    Gy_int = scysparse.csr_matrix((ncv,ncv))  # Matrix to calculate Y-gradient at CVs
    
    nfa_int = np.size(np.where(partofa1=='SOLID')) # Number of internal faces
    
    Avg_cv2f = scysparse.csr_matrix((nfa,ncv)) # Takes gradient from CVs to faces by averaging the values at neighbouring CV values
                
    for i in np.arange(ncv):
        neigh_cv = np.unique(cvofa[faocv[i]]) # Gives neighbouring CVs including the central CV
        neigh_cv = np.delete(neigh_cv,np.where(neigh_cv==i)) # Find index of central CV and delete that entry from neighbouring CV array
        neigh_cv = np.delete(neigh_cv,np.where(neigh_cv==-1)) # Find index of boundary CV and delete the -1 entry from neighbouring CV array
        dx_ik = (xy_cv[neigh_cv,0] - xy_cv[i,0]) # Stores dx for all neighbouring CVs
        dy_ik = (xy_cv[neigh_cv,1] - xy_cv[i,1]) # Stores dy for all neighbouring CVs
        w_ik  = 1./np.sqrt((dx_ik**2) + (dy_ik**2)) # Array of weights for least-squared fit
        a_ik = sum((w_ik*dx_ik)**2) 
        b_ik = sum(((w_ik)**2)*dx_ik*dy_ik)  #Co-efficients a_ik, b_ik, c_ik from least-squared fitting algorithm.
        c_ik = sum((w_ik*dy_ik)**2)
        
        det = (a_ik*c_ik) - (b_ik**2)       
        
        # Filling out weights for collocation point
        Gx_int[i,i] -= sum(((c_ik*((w_ik)**2)*dx_ik) - (b_ik*((w_ik)**2)*dy_ik))/det)
        Gy_int[i,i] -= sum(((a_ik*((w_ik)**2)*dy_ik) - (b_ik*((w_ik)**2)*dx_ik))/det)
        
        for j,n in enumerate(neigh_cv):
            Gx_int[i,n] += ((c_ik*((w_ik[j])**2)*dx_ik[j]) - (b_ik*((w_ik[j])**2)*dy_ik[j]))/det
            Gy_int[i,n] += ((a_ik*((w_ik[j])**2)*dy_ik[j]) - (b_ik*((w_ik[j])**2)*dx_ik[j]))/det
            
    for ii in np.arange(nfa):
        if partofa1[ii]=='SOLID':
            cvs = cvofa[ii]
            Avg_cv2f[ii,cvs] = 0.5
        
    Gx_cvfa = Avg_cv2f*Gx_int
    Gy_cvfa = Avg_cv2f*Gy_int
            
    return (Gx_cvfa,Gy_cvfa)
    
def Divergence_facv(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa):
    
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    faono1 = np.array(faono) # Converting faono to an array
    cold = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'HOT'
    solid = np.where(partofa1=='SOLID')   # Vectorized approach to find face belonging to part 'SOLID'
    
    # Find part to which each node belongs to
    partono = []
    for j in np.arange(nno):
        part_name = partofa1[faono1[j]]
        cold_count = np.array(np.where(part_name == 'COLD')).size
        if cold_count != 0:
            partono.append('COLD')
        else:
            partono.append('SOLID')
    
    nfa_int = np.size(np.where(partofa1=='SOLID'))
    # Divergence operator
    Dx_f2cv = scysparse.csr_matrix((ncv,nfa),dtype="float64") #Creating x part of operator
    Dy_f2cv = scysparse.csr_matrix((ncv,nfa),dtype="float64") #Creating y part of operator
    
    NORMAL = [] #blank normal array to be filled up
    AREA = np.zeros(ncv)
    
    #Pre-processing and finding normals over all CVs
    for i in np.arange(ncv):
        nocv = noofa[faocv[i]]   # Nodal pairs for each face of the CV
        face_co = xy_fa[faocv[i]] # Face centroids of each face of CV
        check_vecs = face_co - xy_cv[i] #Vectors from CV centre to face centre
        par_x = xy_no[nocv[:,1],0] - xy_no[nocv[:,0],0] #x-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        par_y = xy_no[nocv[:,1],1] - xy_no[nocv[:,0],1] #y-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        normal_fa = np.c_[-par_y,par_x]  #Defining normal vector to faces. Convention, normal is 90* clock-wise.
        dir_check = normal_fa[:,0]*check_vecs[:,0] + normal_fa[:,1]*check_vecs[:,1] # Checks if normal_fa is aligned in the same direction as check_vecs.
        normal_fa[np.where(dir_check<0)] = -normal_fa[np.where(dir_check<0)] # Flips sign of components in normal_fa where the dot product i.e. dir_check is negative
        NORMAL.append(normal_fa) # Spits out all normals indexed by Cvs
        
        #Calculating areas of CV assuming rectangles or triangles
        if np.size(faocv[i]) == 3:
            area_cv = np.abs(0.5*((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])))
            AREA[i] = area_cv
            
        if np.size(faocv[i]) == 4:
            area_cv = max(np.abs((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])),np.abs((par_x[0]*par_y[2]) - (par_x[2]*par_y[0])))
            AREA[i] = area_cv
    
    for j in np.arange(ncv):
        normal = NORMAL[j] # Normals of the CV
        #Works as there are utmost 4 nodes right now. Dont know how slow it will be for higher order element shapes
        for ii,nn in enumerate(faocv[j]):
            Dx_f2cv[j,nn] += normal[ii,0]/AREA[j]
            Dy_f2cv[j,nn] += normal[ii,1]/AREA[j]
                
    return (Dx_f2cv,Dy_f2cv)
    
def Avg_nfa_int(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa):
    
    partofa1 = np.array(partofa) # Converting partofa to an array
    faono1 = np.array(faono) # Converting faono to an array
    
    nno = xy_no.shape[0]  # No. of CVs  
    
    nfa = xy_fa.shape[0]
    
    Avg_n2f = scysparse.csr_matrix((nfa,nno)) # Takes gradient from nodes to faces by averaging the values at neighbouring nodes
            
    for ii in np.arange(nfa):
        nodes = noofa[ii]
        Avg_n2f[ii,nodes] = 0.5
            
    return (Avg_n2f)
    
def Avg_fan_int(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa):
    
    nno = xy_no.shape[0]  # No. of CVs  
    
    nfa = xy_fa.shape[0]
    
    partofa1 = np.array(partofa) # Converting partofa to an array
    faono1 = np.array(faono) # Converting faono to an array
    
    cold = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'HOT'
    solid = np.where(partofa1=='SOLID')   # Vectorized approach to find face belonging to part 'SOLID'
    
    partono = []
    for j in np.arange(nno):
        part_name = partofa1[faono1[j]]
        cold_count = np.array(np.where(part_name == 'LID')).size
        if cold_count != 0:
            partono.append('COLD')
        else:
            partono.append('SOLID')
            
    nno_int = nno-np.unique(noofa[cold]).size # No. of internal nodes
    
    Avg_f2n = scysparse.csr_matrix((nno,nfa)) # Takes gradient from nodes to faces by averaging the values at neighbouring nodes
            
    for ii in np.arange(nno_int):
        faces = faono[ii]
        Avg_f2n[ii,faces] = 1./(np.size(faces))
            
    return (Avg_f2n)
    
def Gradient_cvno(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa):
    
    partofa1 = np.array(partofa) # Converting partofa to an array
    faono1 = np.array(faono) # Converting faono to an array
    
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]
    nno = xy_no.shape[0]    
    
    Gx_int = scysparse.csr_matrix((ncv,ncv))  # Matrix to calculate X-gradient at CVs
    Gy_int = scysparse.csr_matrix((ncv,ncv))  # Matrix to calculate Y-gradient at CVs
    
    Avg_cv2no = scysparse.csr_matrix((nno,ncv)) # Takes gradient from CVs to faces by averaging the values at neighbouring CV values
                
    for i in np.arange(ncv):
        neigh_cv = np.unique(cvofa[faocv[i]]) # Gives neighbouring CVs including the central CV
        neigh_cv = np.delete(neigh_cv,np.where(neigh_cv==i)) # Find index of central CV and delete that entry from neighbouring CV array
        neigh_cv = np.delete(neigh_cv,np.where(neigh_cv==-1)) # Find index of boundary CV and delete the -1 entry from neighbouring CV array
        dx_ik = (xy_cv[neigh_cv,0] - xy_cv[i,0]) # Stores dx for all neighbouring CVs
        dy_ik = (xy_cv[neigh_cv,1] - xy_cv[i,1]) # Stores dy for all neighbouring CVs
        w_ik  = 1./np.sqrt((dx_ik**2) + (dy_ik**2)) # Array of weights for least-squared fit
        a_ik = sum((w_ik*dx_ik)**2) 
        b_ik = sum(((w_ik)**2)*dx_ik*dy_ik)  #Co-efficients a_ik, b_ik, c_ik from least-squared fitting algorithm.
        c_ik = sum((w_ik*dy_ik)**2)
        
        det = (a_ik*c_ik) - (b_ik**2)       
        
        # Filling out weights for collocation point
        Gx_int[i,i] -= sum(((c_ik*((w_ik)**2)*dx_ik) - (b_ik*((w_ik)**2)*dy_ik))/det)
        Gy_int[i,i] -= sum(((a_ik*((w_ik)**2)*dy_ik) - (b_ik*((w_ik)**2)*dx_ik))/det)
        
        for j,n in enumerate(neigh_cv):
            Gx_int[i,n] += ((c_ik*((w_ik[j])**2)*dx_ik[j]) - (b_ik*((w_ik[j])**2)*dy_ik[j]))/det
            Gy_int[i,n] += ((a_ik*((w_ik[j])**2)*dy_ik[j]) - (b_ik*((w_ik[j])**2)*dx_ik[j]))/det
            
        faono1 = np.array(faono) # Converting faono to an array
    

    cold = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'HOT'
    solid = np.where(partofa1=='SOLID')   # Vectorized approach to find face belonging to part 'SOLID'
    
    
    cold_bc = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'HOT'
    solid = np.where(partofa1=='SOLID')   # Vectorized approach to find face belonging to part 'SOLID'
    
    partono = []
    for j in np.arange(nno):
        part_name = partofa1[faono1[j]]
        cold_count = np.array(np.where(part_name == 'COLD')).size
        if cold_count != 0:
            partono.append('COLD')
        else:
            partono.append('SOLID')
            
    nno_int = nno-np.unique(noofa[cold_bc]).size # No. of internal nodes
            
    for ii in np.arange(nno_int):
        cvs = np.unique(cvofa[faono[ii]])
        Avg_cv2no[ii,cvs] = 0.5
        
    Gx_cvfa = Avg_cv2no*Gx_int
    Gy_cvfa = Avg_cv2no*Gy_int
            
    return (Gx_cvfa,Gy_cvfa)
    
    