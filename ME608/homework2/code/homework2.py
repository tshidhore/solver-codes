"""
Created on Tue Sep 26 16:13:55 2017

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
import test_script
from scipy.interpolate import griddata
from timeit import default_timer

plt.close('all')

# Gauss Seidel function
def Gauss_Siedel(filename,max_it,tol,omega=1.):
    
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
    
    #Temperature for internal nodes
    phi_int = np.zeros(nno-np.unique(noofa[cold_bc]).size-np.unique(noofa[hot_bc]).size) # Define zero internal nodes
    nno_int = phi_int.size # No. of internal nodes
    
    # Initializing values for hot and cold boundary nodes
    phi_cold = 300 
    phi_hot = 500
    
    #Defining boundary values in phi
    phi[[np.unique(noofa[cold_bc])]] = phi_cold
    phi[[np.unique(noofa[hot_bc])]] = phi_hot
    
    source = -1.
    A = scysparse.csr_matrix((nno_int,nno_int)) # Thank God that boundary nodes get numbered towards the end!
    
    b = source*np.ones(nno_int)

    for i in np.arange(nno_int):
        n_nodes = np.unique(noofa[faono[i]]) # Neighbouring nodes for the collocation point
        x_stencil = xy_no[n_nodes,0] # X-co-ordinates of neighbouring nodes of ith node. All of them have been taken as stencil
        y_stencil = xy_no[n_nodes,1] # Y-co-ordinates of neighbouring nodes of ith node. All of them have been taken as stencil
        xc = xy_no[i,0] # X-co-ordinates of centroid
        yc = xy_no[i,1] # Y-co-ordinates of centroid
        
        # Weights for 2nd derivative for all stencil points
        weights_dx2 = np.zeros(len(n_nodes))
        weights_dy2 = np.zeros(len(n_nodes))
        for ino in range(0,len(n_nodes)):
            phi_base = np.zeros(len(n_nodes))
            phi_base[ino] = 1.0
            _,_,weights_dx2[ino] = fit.BiVarPolyFit_X(xc,yc,x_stencil,y_stencil,phi_base)
            _,_,weights_dy2[ino] = fit.BiVarPolyFit_Y(xc,yc,x_stencil,y_stencil,phi_base)
            
        parts = partono1[n_nodes]
        for jj,node in enumerate(n_nodes):
            if parts[jj] == 'COLD':
                b[i] -= phi_cold*(weights_dx2[jj] + weights_dy2[jj])
            elif parts[jj] == 'HOT':
                b[i] -= phi_hot*(weights_dx2[jj] + weights_dy2[jj])
            else:
                A[i,node] += weights_dx2[jj] + weights_dy2[jj]    
    
    #Define A1 and A2 in A1(phi)^k+1 = A2(phi)^k + q
    A1 = scysparse.tril(A)
    A2 = -scysparse.triu(A,k=1)
    it = 1
    
    #Residual 
    var = 1
    r_0 = np.linalg.norm(b)
    residual = np.ones(1)

    phi_old = 300*np.ones(b.shape)
    #Specified tolerence for r_k/r_0
    print r"\omega = %2.1f" %(omega)
    print "Maximum number of iterations = %d" %(max_it)
    while var and it < max_it:
        Q = (A2*phi_old) + b
        phi_star = splinalg.spsolve(A1,Q)
#        phi_star = np.dot(scysparse.linalg.inv(A1),Q)
        phi = (omega*phi_star) + ((1-omega)*phi_old)
        phi_old = phi
        r_k = np.linalg.norm(b - (A*phi)) #Vector norm of error

        print "Iteration: %d" %(it)
        print "Scaled residual: %2.14f" %(r_k/r_0)
        it += 1
        if (r_k/r_0) < tol:
            residual = np.concatenate([residual,[r_k]])           
            
            break
        
        elif (np.isinf(r_k)==True):
            print "Iterative Solver failed to converge.."
            
            break
        
        residual = np.concatenate([residual,[r_k]])
    
    return A1, A2, phi, it, residual

# Function definitions for the gradient
def analytical_f(x,y):
    return (x*y)**2
    
def grad_x(x,y):
    return (2*x)*(y**2)
    
def grad_y(x,y):
    return (2*y)*(x**2)
    
# hard analytical proof solution
def circle_soln(x,y):
    return (1-(x**2)-(y**2))/4.
    
# Bivariate fit validation
def pol_validation(mesh_no,max_nn):
    
    max_neighb_nodes = np.max(max_nn)
    
    # Explicit check of fit
    for j in np.arange(4,max_neighb_nodes,1):
        test_script.test(j,mesh_no)

# Analytical hard validation for Poisson solver for circle
def Circle_validation(filename):       
 
    mshfile_fullpath = icemcfd_project_folder + filename
    
    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)
    
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    faono1 = np.array(faono) # Converting faono to an array
    hot_bc = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'HOT'
    solid = np.where(partofa1=='SOLID')   # Vectorized approach to find face belonging to part 'SOLID'
    
    x_slice_1 = np.linspace(0.5,0.7,51)
    y_slice_1 = np.linspace(0.1,0.4,51)
    
    # Find part to which each node belongs to
    partono = []
    for j in np.arange(nno):
        part_name = partofa1[faono1[j]]
        hot_count = np.array(np.where(part_name == 'COLD')).size
        if hot_count != 0:
            partono.append('COLD')
        else:
            partono.append('SOLID')
            
    partono1 = np.array(partono)
    
    phi = np.zeros(nno) #Full phi
    
    #Temperature for internal nodes
    phi_int = np.zeros(nno-np.unique(noofa[hot_bc]).size) # Define zero internal nodes
    nno_int = phi_int.size # No. of internal nodes
    
    # Initializing values for hot and cold boundary nodes
    phi_hot = 0.
    
    #Defining boundary values in phi
    phi[[np.unique(noofa[hot_bc])]] = phi_hot
    
    source = -1.
    
    A = scysparse.csr_matrix((nno_int,nno_int)) # Thank God that boundary nodes get numbered towards the end!
    
    b = source*np.ones(nno_int)
    
    for i in np.arange(nno_int):
        n_nodes = np.unique(noofa[faono[i]]) # Neighbouring nodes for the collocation point
        x_stencil = xy_no[n_nodes,0] # X-co-ordinates of neighbouring nodes of ith node. All of them have been taken as stencil
        y_stencil = xy_no[n_nodes,1] # Y-co-ordinates of neighbouring nodes of ith node. All of them have been taken as stencil
        xc = xy_no[i,0] # X-co-ordinates of centroid
        yc = xy_no[i,1] # Y-co-ordinates of centroid
        
        # Weights for 2nd derivative for all stencil points
        weights_dx2 = np.zeros(len(n_nodes))
        weights_dy2 = np.zeros(len(n_nodes))
        for ino in range(0,len(n_nodes)):
            phi_base = np.zeros(len(n_nodes))
            phi_base[ino] = 1.0
            _,_,weights_dx2[ino] = fit.BiVarPolyFit_X(xc,yc,x_stencil,y_stencil,phi_base)
            _,_,weights_dy2[ino] = fit.BiVarPolyFit_Y(xc,yc,x_stencil,y_stencil,phi_base)
            
        parts = partono1[n_nodes]
        for jj,node in enumerate(n_nodes):
            if parts[jj] == 'COLD':
                b[i] -= phi_hot*(weights_dx2[jj] + weights_dy2[jj])
            else:
                A[i,node] += weights_dx2[jj] + weights_dy2[jj]
        
    phi_int = splinalg.spsolve(A,b)
    phi[:nno_int] = phi_int
#    plot_data.plot_data(xy_no[:,0],xy_no[:,1],np.abs(phi - circle_soln(xy_no[:,0],xy_no[:,1])),"Circle: Contour for error in solution.pdf")
    
#    e_RMS = np.sqrt(np.average((phi - circle_soln(xy_no[:,0],xy_no[:,1]))))
    slice_1 = griddata(np.vstack((xy_no[:,0].flatten(),xy_no[:,1].flatten())).T, np.vstack(phi.flatten()),(x_slice_1,y_slice_1),method="cubic")
    return slice_1  
        
def poisson_solver_nc_ds(filename,mesh_no,flag=0):
    
    mshfile_fullpath = icemcfd_project_folder + filename
    
    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)
    
    # Two slices
    x_slice_1 = np.linspace(1.0,1.5,51)
    y_slice_1 = np.linspace(0.1,0.4,51)

    x_slice_2 = 1.5*np.ones(51)
    y_slice_2 = np.linspace(0.3,0.8,51)      
    
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
    
    #Temperature for internal nodes
    phi_int = np.zeros(nno-np.unique(noofa[cold_bc]).size-np.unique(noofa[hot_bc]).size) # Define zero internal nodes
    nno_int = phi_int.size # No. of internal nodes
    
    # Initializing values for hot and cold boundary nodes
    phi_cold = 300 
    phi_hot = 500
    
    #Defining boundary values in phi
    phi[[np.unique(noofa[cold_bc])]] = phi_cold
    phi[[np.unique(noofa[hot_bc])]] = phi_hot
    
    source = -1.
    
    if mesh_no == 3:
        
        # For bivariate fit validation
        max_nn = np.zeros(nno_int)    
    
    A = scysparse.csr_matrix((nno_int,nno_int)) # Thank God that boundary nodes get numbered towards the end!
    b = source*np.ones(nno_int)
    
    for i in np.arange(nno_int):
        n_nodes = np.unique(noofa[faono[i]]) # Neighbouring nodes for the collocation point
        if mesh_no == 3:
            max_nn[i] = n_nodes.size
        x_stencil = xy_no[n_nodes,0] # X-co-ordinates of neighbouring nodes of ith node. All of them have been taken as stencil
        y_stencil = xy_no[n_nodes,1] # Y-co-ordinates of neighbouring nodes of ith node. All of them have been taken as stencil
        xc = xy_no[i,0] # X-co-ordinates of centroid
        yc = xy_no[i,1] # Y-co-ordinates of centroid
        
        # Weights for 2nd derivative for all stencil points
        weights_dx2 = np.zeros(len(n_nodes))
        weights_dy2 = np.zeros(len(n_nodes))
        for ino in range(0,len(n_nodes)):
            phi_base = np.zeros(len(n_nodes))
            phi_base[ino] = 1.0
            _,_,weights_dx2[ino] = fit.BiVarPolyFit_X(xc,yc,x_stencil,y_stencil,phi_base)
            _,_,weights_dy2[ino] = fit.BiVarPolyFit_Y(xc,yc,x_stencil,y_stencil,phi_base)
            
        parts = partono1[n_nodes]
        for jj,node in enumerate(n_nodes):
            if parts[jj] == 'COLD':
                b[i] -= phi_cold*(weights_dx2[jj] + weights_dy2[jj])
            elif parts[jj] == 'HOT':
                b[i] -= phi_hot*(weights_dx2[jj] + weights_dy2[jj])
            else:
                A[i,node] += weights_dx2[jj] + weights_dy2[jj]
                
    if mesh_no == 3:
        pol_validation(mesh_no,max_nn)
    
    start_time1 = default_timer()
    phi_int = splinalg.spsolve(A,b)
    end_time1 = default_timer() - start_time1
    phi[:nno_int] = phi_int
    plot_data.plot_data(xy_no[:,0],xy_no[:,1],phi,"Mesh " + str(mesh_no) + ": Final Temperature Field Direct Solve.pdf")
    
    plt.spy(A)
    plt.savefig(figure_folder + "Mesh " + str(mesh_no) + ": Spy of A.pdf")
    plt.close()
    
    slice_1 = griddata(np.vstack((xy_no[:,0].flatten(),xy_no[:,1].flatten())).T, np.vstack(phi.flatten()),(x_slice_1,y_slice_1),method="cubic")
    slice_2 = griddata(np.vstack((xy_no[:,0].flatten(),xy_no[:,1].flatten())).T, np.vstack(phi.flatten()),(x_slice_2,y_slice_2),method="cubic")
    
    if flag==1:
        
        print "The time required for execution of spsolve is: 2.10f" %(end_time1)
        return(A)
        
    else:
        
        return(A,phi,slice_1,slice_2)
        
def Gradient(filename,mesh_no,flag=0):
    
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
    
    Gx_int = scysparse.csr_matrix((ncv,ncv))  # Matrix to calculate X-gradient at CVs
    Gy_int = scysparse.csr_matrix((ncv,ncv))  # Matrix to calculate Y-gradient at CVs
    
    nfa_int = np.size(np.where(partofa1=='SOLID'))
    Avg_cv2f = scysparse.csr_matrix((nfa_int,ncv))
                
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

    for ii in np.arange(nfa_int):
        cvs = cvofa[ii]
        Avg_cv2f[ii,cvs] = 0.5
    
    Gx = Avg_cv2f*Gx_int
    Gy = Avg_cv2f*Gy_int
    if flag==1:
        # Validation of gradient evaluation
        
        phi_cv = analytical_f(xy_cv[:,0],xy_cv[:,1])
        
        grad_phi_analytical_x = grad_x(xy_fa[:nfa_int,0],xy_fa[:nfa_int,1])
        
        grad_phi_analytical_y = grad_y(xy_fa[:nfa_int,0],xy_fa[:nfa_int,1])
        
        grad_phi_num_x = Gx*phi_cv
        grad_phi_num_y = Gy*phi_cv
        
        
        plt.quiver(xy_fa[:nfa_int,0],xy_fa[:nfa_int,1],grad_phi_analytical_x,grad_phi_analytical_y,color='b')
        plt.quiver(xy_fa[:nfa_int,0],xy_fa[:nfa_int,1],grad_phi_num_x,grad_phi_num_y,color='r')
        print "Saving figure: "+ figure_folder + "Mesh"+ str(mesh_no) + "Quiver plot for gradient.pdf"
        plt.savefig(figure_folder + "Mesh"+ str(mesh_no) + "Quiver_gradient.pdf")
        plt.close()
    
        plt.spy(Gx)
        plt.savefig(figure_folder + "Mesh " + str(mesh_no) + ": Spy of Gx.pdf")
        plt.close()
    
        plt.spy(Gy)
        plt.savefig(figure_folder + "Mesh " + str(mesh_no) + ": Spy of Gy.pdf")
        plt.close()      
    
    return (Gx,Gy)
    
def Divergence_val(filename,mesh_no):
    
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
    
    nfa_int = np.size(np.where(partofa1=='SOLID'))
        # Divergence operator
    Dx_f2cv = scysparse.csr_matrix((ncv,nfa_int),dtype="float64") #Creating x part of operator
    Dy_f2cv = scysparse.csr_matrix((ncv,nfa_int),dtype="float64") #Creating y part of operator
    
    q_bc = np.zeros(ncv)
    u_bc = np.zeros(nfa)
    v_bc = np.zeros(nfa)
    
    u = (xy_fa[:,1])*(xy_fa[:,0]**2) #x-component of velocity
    v = -(xy_fa[:,0])*(xy_fa[:,1]**2) #y-component of velocity
        
    for l in np.arange(nfa):
        if partofa1[l] != 'SOLID':
            u_bc[l] = u[l]
            v_bc[l] = v[l]
    
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
            if partofa1[nn] == 'SOLID':
                Dx_f2cv[j,nn] += normal[ii,0]/AREA[j]
                Dy_f2cv[j,nn] += normal[ii,1]/AREA[j]
                
            else:
                q_bc[j] += u_bc[nn]*normal[ii,0]/AREA[j] + v_bc[nn]*normal[ii,1]/AREA[j]
                
    
        
    DIVERGENCE = (Dx_f2cv.dot(u[:nfa_int])) + (Dy_f2cv.dot(v[:nfa_int])) + q_bc
    
    e_RMS = np.sqrt(np.average(DIVERGENCE**2))
    
    plot_data.plot_data(xy_cv[:,0],xy_cv[:,1],DIVERGENCE,"Mesh " + str(mesh_no) + "Flooded_Contour_of_Divergence.pdf")
                
    plt.spy(Dx_f2cv)
    plt.savefig(figure_folder + "Mesh " + str(mesh_no) + ": Spy of Dx.pdf")
    plt.close()

    plt.spy(Dy_f2cv)
    plt.savefig(figure_folder + "Mesh " + str(mesh_no) + ": Spy of Dy.pdf")
    plt.close()      
                
    return (Dx_f2cv,Dy_f2cv,q_bc,e_RMS)
    
def correction(filename,mesh_no):
    
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
    
    nfa_int = np.size(np.where(partofa1=='SOLID'))
        # Divergence operator
    Dx_f2cv = scysparse.csr_matrix((ncv,nfa_int),dtype="float64") #Creating x part of operator
    Dy_f2cv = scysparse.csr_matrix((ncv,nfa_int),dtype="float64") #Creating y part of operator
    
    q_bc = np.zeros(ncv)
    u_bc = np.zeros(nfa)
    v_bc = np.zeros(nfa)
    
    u = (xy_fa[:,1])*(xy_fa[:,0]**2) #x-component of velocity
    u_star = np.zeros(u.size)
    u_star[:nfa_int] = (xy_fa[:nfa_int,1])*(xy_fa[:nfa_int,0]**2) + 0.1*xy_fa[:nfa_int,0]
    u_star[nfa_int:] = u[nfa_int:]
    u_corr = np.zeros(u.size)
    v = -(xy_fa[:,0])*(xy_fa[:,1]**2) #y-component of velocity
    v_star = np.zeros(v.size)
    v_star[:nfa_int] = (xy_fa[:nfa_int,1])*(xy_fa[:nfa_int,0]**2) + 0.1*xy_fa[:nfa_int,0]
    v_star[nfa_int:] = v[nfa_int:]
    v_corr = np.zeros(v.size)
        
    for l in np.arange(nfa):
        if partofa1[l] != 'SOLID':
            u_bc[l] = u[l]
            v_bc[l] = v[l]
    
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
            if partofa1[nn] == 'SOLID':
                Dx_f2cv[j,nn] += normal[ii,0]/AREA[j]
                Dy_f2cv[j,nn] += normal[ii,1]/AREA[j]
                
            else:
                q_bc[j] += u_bc[nn]*normal[ii,0]/AREA[j] + v_bc[nn]*normal[ii,1]/AREA[j]
                
    Gx,Gy = Gradient(filename,mesh_no)
    
    A = (Dx_f2cv.dot(Gx)) + (Dy_f2cv.dot(Gy))
    
    dt = 1.
    
    RHS = ((Dx_f2cv.dot(u_star[:nfa_int])) + (Dy_f2cv.dot(v_star[:nfa_int])) + q_bc)/dt
    
    Div_starred = RHS*dt

    rank = np.linalg.matrix_rank(A.todense())

    A[0,:] = 1.    
    
    phi = splinalg.spsolve(A,RHS)
    
    u_corr[:nfa_int] = u_star[:nfa_int] - ((dt)*Gx.dot(phi))
    
    v_corr[:nfa_int] = v_star[:nfa_int] - ((dt)*Gy.dot(phi))
    
    u_corr[nfa_int:] = u_star[nfa_int:]
    
    v_corr[nfa_int:] = v_star[nfa_int:]
    
    Div_corr = (Dx_f2cv.dot(u_corr[:nfa_int])) + (Dy_f2cv.dot(v_corr[:nfa_int])) + q_bc
    
    plt.quiver(xy_fa[:,0],xy_fa[:,1],u_corr,v_corr,color='b')
    plt.quiver(xy_fa[:,0],xy_fa[:,1],u,v,color='r')
    print "Saving figure: "+ figure_folder + "Mesh"+ str(mesh_no) + "Quiver plot for corrected and analytical.pdf"
    plt.savefig(figure_folder + "Mesh"+ str(mesh_no) + "Quiver_velocity.pdf")
    plt.close()
    
    plt.quiver(xy_fa[:,0],xy_fa[:,1],u_corr,v_corr,color='b')
    print "Saving figure: "+ figure_folder + "Mesh"+ str(mesh_no) + "Quiver plot for starred velocity.pdf"
    plt.savefig(figure_folder + "Mesh"+ str(mesh_no) + "Quiver_velocity_star.pdf")
    plt.close()
    
    plot_data.plot_data(xy_cv[:,0],xy_cv[:,1],np.log(np.abs(Div_starred)),"Mesh " + str(mesh_no) + "Flooded_Contour_of_divergence_perturbed_field.pdf")
    plot_data.plot_data(xy_cv[:,0],xy_cv[:,1],np.log(np.abs(Div_corr)),"Mesh " + str(mesh_no) + "Flooded_Contour_of_divergence_corrected_field.pdf")
    
    e_RMS = np.sqrt(np.average((Div_corr-np.zeros(Div_corr.size))**2))
    
    return (rank,e_RMS)

p1 = False
p2 = False
pcirc = False
p3 = True

figure_folder = "../report/figures/"
icemcfd_project_folder = './mesh_folder/'

figwidth,figheight = 14,12
lineWidth = 3
fontSize = 25
gcafontSize = 21

if p1:
    
    filename = "Mesh_1.msh"
    mesh_no = 1
    A,phi,slice_1_1,slice_2_1 = poisson_solver_nc_ds(filename,mesh_no)
    
    filename = "Mesh_2.msh"
    mesh_no = 2
    A,phi,slice_1_2,slice_2_2 = poisson_solver_nc_ds(filename,mesh_no)
    
    filename = "Mesh_3.msh"
    mesh_no = 3
    A,phi,slice_1_3,slice_2_3 = poisson_solver_nc_ds(filename,mesh_no)
    
    filename = "Mesh_4.msh"
    mesh_no = 4
    A,phi,slice_1_4,slice_2_4 = poisson_solver_nc_ds(filename,mesh_no)
    
    # Plots for grid convergence
    figwidth,figheight = 14,12
    fig = plt.figure(1,(figwidth,figheight))
    ax = fig.add_subplot(2,1,1)
    ax.plot(slice_1_1,label='Mesh 1')
    ax.plot(slice_1_2,label='Mesh 2')
    ax.plot(slice_1_3,label='Mesh 3')
    ax.plot(slice_1_4,label='Mesh 4')
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.set_xlabel("Slice index",fontsize=fontSize)
    ax.set_ylabel(r"$\phi$",fontsize=fontSize)
    ax.grid(True)
    
    ax = fig.add_subplot(2,1,2)
    ax.plot(slice_2_1,label='Mesh 1')
    ax.plot(slice_2_2,label='Mesh 2')
    ax.plot(slice_2_3,label='Mesh 3')
    ax.plot(slice_2_4,label='Mesh 4')
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.set_xlabel("Slice index",fontsize=fontSize)
    ax.set_ylabel(r"$\phi$",fontsize=fontSize)
    ax.grid(True)
    
    plt.legend(loc='best',fontsize=gcafontSize)
    print "Saving figure: "+ figure_folder + "Grid Convergence.pdf"
    plt.savefig(figure_folder + "Grid Convergence.pdf")
    plt.close()    
   
if pcirc:
    # Hard validation for the circle
    filename = "circle_tria_1.msh"
    slice1_circ = Circle_validation(filename)
    filename = "circle_tria_2.msh"
    slice2_circ = Circle_validation(filename)
    filename = "circle_tria_3.msh"
    slice3_circ = Circle_validation(filename)
    filename = "circle_tria_4.msh"
    slice4_circ = Circle_validation(filename)
    
    x_slice_1 = np.linspace(0.5,0.7,51)
    y_slice_1 = np.linspace(0.1,0.4,51)
    solution = circle_soln(x_slice_1,y_slice_1)
    figwidth,figheight = 14,12
    fig = plt.figure(1,(figwidth,figheight))
    plt.plot(slice1_circ,label='Mesh 1')
    plt.plot(slice2_circ,label='Mesh 2')
    plt.plot(slice3_circ,label='Mesh 3')
    plt.plot(slice4_circ,label='Mesh 4')
    plt.plot(solution,label='Analytical')
#    plt.setp(plt.xticklabels(),fontsize=gcafontSize)
#    plt.setp(plt.yticklabels(),fontsize=gcafontSize)
    plt.xlabel("Slice index",fontsize=fontSize)
    plt.ylabel(r"$\phi$",fontsize=fontSize)
    plt.grid(True)
    
    plt.legend(loc='best',fontsize=gcafontSize)
    print "Saving figure: "+ figure_folder + "Grid Convergence Circle.pdf"
    plt.savefig(figure_folder + "Grid Convergence Circle.pdf")
    plt.close() 
 
if p2:
    
    max_it = 1000
    tol = 10**(-7)
    
    omega = np.linspace(0.1,2.,15)
    eig_1 = np.zeros(omega.size)
    eig_2 = np.zeros(omega.size)    
    eig_3 = np.zeros(omega.size)    
    eig_4 = np.zeros(omega.size)
        
    filename = "Mesh_1.msh"
    A11,A21,phi_1,it1,res_1  = Gauss_Siedel(filename,max_it,tol)
    B = splinalg.spsolve(A11,A21)
    eig,_ = splinalg.eigs(B,k=1,which='LM')
    print "For the first bad mesh, spectral radius of B = %f" %(eig[0])
    I = np.eye(B.shape[0])
    for jj,omeg in enumerate(omega):
        
        Mod_matrix = ((omeg)*B) + ((1-omeg)*I)
        eig,_ = splinalg.eigs(Mod_matrix,k=1,which='LM')
        eig_1[jj] = eig[0]
    
    filename = "Mesh_2.msh"
    A11,A21,phi_1,it1,res_1  = Gauss_Siedel(filename,max_it,tol)
    B = splinalg.spsolve(A11,A21)
    eig,_ = splinalg.eigs(B,k=1,which='LM')
    print "For the second bad mesh, spectral radius of B = %f" %(eig[0])
    I = np.eye(B.shape[0])
    for jj,omeg in enumerate(omega):
        
        Mod_matrix = ((omeg)*B) + ((1-omeg)*I)
        eig,_ = splinalg.eigs(Mod_matrix,k=1,which='LM')
        eig_2[jj] = eig[0]
    
    filename = "Mesh_3.msh"
    A11,A21,phi_1,it1,res_1  = Gauss_Siedel(filename,max_it,tol)
    B = splinalg.spsolve(A11,A21)
    eig,_ = splinalg.eigs(B,k=1,which='LM')
    print "For the thrid bad mesh, spectral radius of B = %f" %(eig[0])
    I = np.eye(B.shape[0])
    for jj,omeg in enumerate(omega):
        
        Mod_matrix = ((omeg)*B) + ((1-omeg)*I)
        eig,_ = splinalg.eigs(Mod_matrix,k=1,which='LM')
        eig_3[jj] = eig[0]
    
    filename = "Mesh_4.msh"
    A11,A21,phi_1,it1,res_1  = Gauss_Siedel(filename,max_it,tol)
    B = splinalg.spsolve(A11,A21)
    eig,_ = splinalg.eigs(B,k=1,which='LM')
    print "For the fourth bad mesh, spectral radius of B = %f" %(eig[0])
    I = np.eye(B.shape[0])
    for jj,omeg in enumerate(omega):
        
        Mod_matrix = ((omeg)*B) + ((1-omeg)*I)
        eig,_ = splinalg.eigs(Mod_matrix,k=1,which='LM')
        eig_4[jj] = eig[0]
    
    fig = plt.figure(0,(figwidth,figheight))
    plt.plot(omega,eig_1,label='Mesh 1')
    plt.plot(omega,eig_2,label='Mesh 2')
    plt.plot(omega,eig_3,label='Mesh 3')
    plt.plot(omega,eig_4,label='Mesh 4')
    plt.xticks(fontsize=gcafontSize)
    plt.yticks(fontsize=gcafontSize)
    plt.xlabel(r"$\omega$",fontsize=fontSize)
    plt.ylabel("Spectral radius for modified iteration matrix",fontsize=fontSize)
    plt.legend(loc='best',fontsize=gcafontSize)
    print "Saving figure: "+ figure_folder + "Omega_optimization_C.pdf"
    plt.savefig(figure_folder + "Omega_optimization_C.pdf")
    plt.close()
    
#     Timing spsolve on four bad meshes
    
    filename = "Bad_Mesh_1.msh"
    mesh_no = 5
    poisson_solver_nc_ds(filename,mesh_no,1)
    
    filename = "Bad_Mesh_2.msh"
    mesh_no = 6
    poisson_solver_nc_ds(filename,mesh_no,1)
    
    filename = "Bad_Mesh_3.msh"
    mesh_no = 7
    poisson_solver_nc_ds(filename,mesh_no,1)
    
    filename = "Bad_Mesh_4.msh"
    mesh_no = 8
    t4 = poisson_solver_nc_ds(filename,mesh_no,1)
    print " The time required to execute spsolve on the the first bad mesh is %f" %(t4)
    
if p3:
    
    filename = "Mesh_1.msh"
    mesh_no = 1
    Gx1,Gy1 = Gradient(filename,mesh_no,1)
    Dx_f2cv1,Dy_f2cv1,q_bc1,e_RMS1 = Divergence_val(filename, mesh_no)
     #%%   
    filename = "Mesh_2.msh"
    mesh_no = 2
    Gx1,Gy1 = Gradient(filename,mesh_no,1)
    Dx_f2cv2,Dy_f2cv2,q_bc2,e_RMS2 = Divergence_val(filename, mesh_no)
    #%%
    filename = "Mesh_3.msh"
    mesh_no = 3
    Gx1,Gy1 = Gradient(filename,mesh_no,1)
    Dx_f2cv3,Dy_f2cv3,q_bc3,e_RMS3 = Divergence_val(filename, mesh_no)
    #%%
    filename = "Mesh_4.msh"
    mesh_no = 4
    Gx1,Gy1 = Gradient(filename,mesh_no,1)
    Dx_f2cv4,Dy_f2cv4,q_bc4,e_RMS4 = Divergence_val(filename, mesh_no)
    #%%
    filename = "Mesh_1.msh"
    mesh_no = 1
    rank1,RMS1 = correction(filename,mesh_no)
    print "Rank of A: %d" %(rank1)
       #%% 
    filename = "Mesh_2.msh"
    mesh_no = 2
    rank2,RMS2 = correction(filename,mesh_no)
    print "Rank of A: %d" %(rank2)
    #%%
    filename = "Mesh_3.msh"
    mesh_no = 3
    rank3,RMS3 = correction(filename,mesh_no)
    print "Rank of A: %d" %(rank3)
    #%%
    filename = "Mesh_4.msh"
    mesh_no = 4
    rank4,RMS4 = correction(filename,mesh_no)
    print "Rank of A: %d" %(rank4)
#    #%%
#    NCV = np.array([rank1+1,rank2+1,rank3+1,rank4+1])
#    RMS_error = np.array([e_RMS1,e_RMS2,e_RMS3,e_RMS4])
#    figwidth,figheight = 14,12
#    fig = plt.figure(1,(figwidth,figheight))
#    plt.loglog(NCV,RMS_error,linewidth=lineWidth)
#    plt.loglog(NCV,1/NCV,'k',label="first order")
#    plt.loglog(NCV,1/(NCV**2),'r',label="first order")
#    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
#    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
#    ax.set_xlabel(r"$1/h$",fontsize=fontSize)
#    ax.set_ylabel(r"RMS error",fontsize=fontSize)
#    ax.grid(True)
#    plt.savefig(figure_folder + "RMS_Divergence.pdf")
#    plt.close()

    
    
