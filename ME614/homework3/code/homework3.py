import os
import sys
import numpy as np
import scipy.sparse as scysparse
from pdb import set_trace as keyboard
from time import sleep
import spatial_operators
import scipy.sparse as scysparse
import scipy.sparse.linalg as spysparselinalg  # sparse linear algebra
import scipy.linalg as scylinalg               # non-sparse linear algebra
import pylab as plt
from matplotlib import rc as matplotlibrc
import time # has the equivalent of tic/toc

machine_epsilon = np.finfo(float).eps
matplotlibrc('text.latex', preamble='\usepackage{color}')
matplotlibrc('text',usetex=True)
matplotlibrc('font', family='serif')

#Gauss-Seidel solver for a particular RHS. RHS need sto be specified in the solver. 
#Giving it as a generic input would be quite tedious and was not done, keeping in mind the specific questions of this homework
def Gauss_Seidel(Nxc,Nyc,Lx,Ly,Re,n,omega,BC,it_max,tol,sd):
    # number of (pressure) cells = mass conservation cells
    #########################################
    ######## Preprocessing Stage ############
    
    # You might have to include ghost-cells here
    # Depending on your application
    
    # define grid for u and v velocity components first
    # and then define pressure cells locations
    xsi_u = np.linspace(0.,1.0,Nxc+1)
    xsi_v = np.linspace(0.,1.0,Nyc+1)
    # uniform grid
    xu = xsi_u*Lx
    yv = xsi_v*Ly
    # non-uniform grid can be specified by redefining xsi_u and xsi_v. For e.g.
    #xu = (xsi_u**4.)*Lx # can be replaced by non-uniform grid
    #yv = (xsi_v**4.)*Ly # can be replaced by non-uniform grid
    
    # creating ghost cells
    #internodal distance at the beginning and end
    dxu0 = np.diff(xu)[0]
    dxuL = np.diff(xu)[-1]
    #Concatenate the ghost points (obtained by flipping the same distances on either side of the first and last point)
    xu = np.concatenate([[xu[0]-dxu0],xu,[xu[-1]+dxuL]])
    
    dyv0 = np.diff(yv)[0]
    dyvL = np.diff(yv)[-1]
    yv = np.concatenate([[yv[0]-dyv0],yv,[yv[-1]+dyvL]])
    
    
    #Pressure nodes are cell centres
    #Velocity nodes described above are staggered wrt pressure nodes.
    dxc = np.diff(xu)  # pressure-cells spacings
    dyc = np.diff(yv)
    
    xc = 0.5*(xu[:-1]+xu[1:])  # total of Nxc cells
    yc = 0.5*(yv[:-1]+yv[1:])  # total of Nyc cells
    
    # note that indexing is Xc[j_y,i_x] or Xc[j,i]
    [Xc,Yc]   = np.meshgrid(xc,yc)     # these arrays are with ghost cells
    [Dxc,Dyc] = np.meshgrid(dxc,dyc)   # these arrays are with ghost cells
    
    ### familiarize yourself with 'flattening' options
    # phi_Fordering     = Phi.flatten(order='F') # F (column-major), Fortran/Matlab default
    # phi_Cordering     = Phi.flatten(order='C') # C (row-major), C/Python default
    # phi_PythonDefault = Phi.flatten()          # Python default
    # compare Phi[:,0] and Phi[0,:] with phi[:Nxc]
    
    #Mask has been flipped as compared to the starter code.
    #Mask is designed to be false for all points where pressure is to be calculated i.e. Mask==True means pressure is masked at those regions
    #This required changing np.zeros to np.ones in line 71
    # Pre-allocated False = fluid points
    pressureCells_Mask = np.ones(Xc.shape)
    pressureCells_Mask[1:-1,1:-1] = False # note that ghost cells are also marked true as a default value of 1 is assigned
    
    # Introducing obstacle in pressure Mask
    obstacle_radius = 0.0*Lx # set equal to 0.0*Lx to remove obstacle
    distance_from_center = np.sqrt(np.power(Xc-Lx/2.,2.0)+np.power(Yc-Ly/2.,2.0))
    j_obstacle,i_obstacle = np.where(distance_from_center<obstacle_radius)
    pressureCells_Mask[j_obstacle,i_obstacle] = True
    
    # number of actual pressure cells
    #Test case scenario, when q=1 uniformly
#    Np = len(np.where(pressureCells_Mask==False)[0]) #Best for storing uniform conditions in q
    #Generate 1D arrays jj_C and ii_C containing all pressure cell indices where mask is false i.e. pressure is calculated
    #Assign q in a flattened (1D) array
    jj_C,ii_C = np.where(pressureCells_Mask==False)
    q  = 2*np.pi*n*np.sin(2*np.pi*n*Yc[jj_C,ii_C])*(np.cos(2*np.pi*n*Xc[jj_C,ii_C]) + (4*np.pi*n*np.sin(2*np.pi*n*Xc[jj_C,ii_C])/Re))
    #Actual solution of the problem (for testing)
    sol = np.sin(2*np.pi*n*Xc[jj_C,ii_C])*np.sin(2*np.pi*n*Yc[jj_C,ii_C])
    #Test case scenario for q=1 uniformly
#    q = np.ones(Np)
    
    # a more advanced option is to separately create the divergence and gradient operators
    #Create laplacian using specified BCs
    DivGrad = spatial_operators.create_DivGrad_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask,BC)
    
    #Create del(phi)/del(x)
    Div = spatial_operators.create_delx_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask,sd,BC)
    # if boundary_conditions are not specified, it defaults to "Homogeneous Neumann"
    #keyboard()
    
    #Core Gauss-Seidel code block
    A = Div - (DivGrad/Re)
    
    #Define A1 and A2 in A1(phi)^k+1 = A2(phi)^k + q
    A1 = scysparse.tril(A)
    A2 = -scysparse.triu(A,k=1)
    it = 1
    #Residual 
    var = 1
    r_0 = np.linalg.norm(q)
    residual = np.ones(1)

    phi_old = np.zeros(q.shape)
    #Specified tolerence for r_k/r_0
    print "Grid size = %d x %d, Re = %f omega = %2.1f" %(Nxc,Nyc,Re,omega)
    print "Wavenumber = %d, maximum number of iterations = %d" %(n,max_it)
    while var and it < max_it:
        Q = (A2*phi_old) + q
        phi_star = spysparselinalg.spsolve(A1,Q)
#        phi_star = np.dot(scysparse.linalg.inv(A1),Q)
        phi = (omega*phi_star) + ((1-omega)*phi_old)
        phi_old = phi
        r_k = np.linalg.norm(q - (A*phi)) #Vector norm of error
#        print np.max(np.abs(phi-sol))
#        print it
#        print r_k
        print "Iteration: %d" %(it)
        print "Scaled residual: %2.14f" %(r_k/r_0)
        it += 1
        if (r_k/r_0) < tol:
            residual = np.concatenate([residual,[r_k]])
            
#            print it            
            
            break
        
        residual = np.concatenate([residual,[r_k]])
       
            
        


    #Solution by direct solve    
    PHI = spysparselinalg.spsolve(A,q)
    
    #Pouring all 1D flattened array (of final solution using Gauss-Siedel, direct solve, and analytical evaluation of phi (sol))
    PHi = np.zeros((Nyc+2,Nxc+2))*np.nan
    PHi[np.where(pressureCells_Mask==False)] = PHI
    PHi_1 = np.zeros((Nyc+2,Nxc+2))*np.nan
    PHi_1[np.where(pressureCells_Mask==False)] = phi
    SOL = np.zeros((Nyc+2,Nxc+2))*np.nan
    SOL[np.where(pressureCells_Mask==False)] = sol
    
    #Plotter code for the contours of all solutions for comparison
#    figwidth       = 10
#    figheight      = 6
#    lineWidth      = 4
#    textFontSize   = 28
#    gcafontSize    = 30
#    
##    figure_name_1 = "Residual history for n=(%d) and w=(%2.3f).pdf" (%n) (%omega)  
#    
#    fig1 = plt.figure(0, figsize=(figwidth,figheight))
#    plt.plot(iteration[1:],residual[1:])
#    plt.setp(plt.xticklabels(),fontsize=gcafontSize)
#    plt.setp(plt.yticklabels(),fontsize=gcafontSize)
#    plt.grid('on',which='both')
##    plt.set_xlabel(r"iteration no. k",fontsize=textFontSize)
##    plt.set_ylabel(r"$r_{k}$",fontsize=textFontSize,rotation=90)
#    plt.title("Residual history for n= (%d), w = (%2.3f)" %n %omega)
#    plt.legend(loc='best')
#    plt.show()
    
#    # Plot solution. For code validation
#    fig = plt.figure(0, figsize=(figwidth,figheight))
#    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#    plt.axes(ax)
#    plt.contourf(Xc,Yc,PHi)
#    plt.colorbar()
#    
#    fig = plt.figure(1, figsize=(figwidth,figheight))
#    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#    plt.axes(ax)
#    plt.contourf(Xc,Yc,SOL)
#    plt.colorbar()
#    
#    fig = plt.figure(2, figsize=(figwidth,figheight))
#    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#    plt.axes(ax)
#    plt.contourf(Xc,Yc,PHi_1)
#    plt.colorbar()
##    
#    ax.grid('on',which='both')
#    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
#    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
#    ax.set_xlabel(r"$x$",fontsize=textFontSize)
#    ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
#    plt.axis("tight")
#    plt.axis("equal")
#    
#    plt.show()
#    
#    fig = plt.figure(1, figsize=(figwidth,figheight))
#    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#    plt.axes(ax)
#    plt.contourf(Xc,Yc,PHi_1)
#    plt.colorbar()
#    
#    ax.grid('on',which='both')
#    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
#    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
#    ax.set_xlabel(r"$x$",fontsize=textFontSize)
#    ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
#    plt.axis("tight")
#    plt.axis("equal")
#    
#    plt.show()
#    
#    fig = plt.figure(2, figsize=(figwidth,figheight))
#    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#    plt.axes(ax)
#    plt.contourf(Xc,Yc,SOL)
#    plt.colorbar()
#    
#    ax.grid('on',which='both')
#    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
#    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
#    ax.set_xlabel(r"$x$",fontsize=textFontSize)
#    ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
#    plt.axis("tight")
#    plt.axis("equal")
#    
#    plt.show()
#        
#    
#    
#    
##    Figure settings
#    matplotlibrc('text.latex', preamble='\usepackage{color}')
#    matplotlibrc('text',usetex=True)
#    matplotlibrc('font', family='serif')
    
    return it, residual

def NLa(Nxc,Nyc,Lx,Ly,Re,n,omega,BC,it_max,tol,sd):
    # number of (pressure) cells = mass conservation cells
    #########################################
    ######## Preprocessing Stage ############
    
    # You might have to include ghost-cells here
    # Depending on your application
    
    # define grid for u and v velocity components first
    # and then define pressure cells locations
    xsi_u = np.linspace(0.,1.0,Nxc+1)
    xsi_v = np.linspace(0.,1.0,Nyc+1)
    # uniform grid
    xu = xsi_u*Lx
    yv = xsi_v*Ly
    # non-uniform grid can be specified by redefining xsi_u and xsi_v. For e.g.
    #xu = (xsi_u**4.)*Lx # can be replaced by non-uniform grid
    #yv = (xsi_v**4.)*Ly # can be replaced by non-uniform grid
    
    # creating ghost cells
    #internodal distance at the beginning and end
    dxu0 = np.diff(xu)[0]
    dxuL = np.diff(xu)[-1]
    #Concatenate the ghost points (obtained by flipping the same distances on either side of the first and last point)
    xu = np.concatenate([[xu[0]-dxu0],xu,[xu[-1]+dxuL]])
    
    dyv0 = np.diff(yv)[0]
    dyvL = np.diff(yv)[-1]
    yv = np.concatenate([[yv[0]-dyv0],yv,[yv[-1]+dyvL]])
    
    
    #Pressure nodes are cell centres
    #Velocity nodes described above are staggered wrt pressure nodes.
    dxc = np.diff(xu)  # pressure-cells spacings
    dyc = np.diff(yv)
    
    xc = 0.5*(xu[:-1]+xu[1:])  # total of Nxc cells
    yc = 0.5*(yv[:-1]+yv[1:])  # total of Nyc cells
    
    # note that indexing is Xc[j_y,i_x] or Xc[j,i]
    [Xc,Yc]   = np.meshgrid(xc,yc)     # these arrays are with ghost cells
    [Dxc,Dyc] = np.meshgrid(dxc,dyc)   # these arrays are with ghost cells
    
    ### familiarize yourself with 'flattening' options
    # phi_Fordering     = Phi.flatten(order='F') # F (column-major), Fortran/Matlab default
    # phi_Cordering     = Phi.flatten(order='C') # C (row-major), C/Python default
    # phi_PythonDefault = Phi.flatten()          # Python default
    # compare Phi[:,0] and Phi[0,:] with phi[:Nxc]
    
    #Mask has been flipped as compared to the starter code.
    #Mask is designed to be false for all points where pressure is to be calculated i.e. Mask==True means pressure is masked at those regions
    #This required changing np.zeros to np.ones in line 71
    # Pre-allocated False = fluid points
    pressureCells_Mask = np.ones(Xc.shape)
    pressureCells_Mask[1:-1,1:-1] = False # note that ghost cells are also marked true as a default value of 1 is assigned
    
    # Introducing obstacle in pressure Mask
    obstacle_radius = 0.0*Lx # set equal to 0.0*Lx to remove obstacle
    distance_from_center = np.sqrt(np.power(Xc-Lx/2.,2.0)+np.power(Yc-Ly/2.,2.0))
    j_obstacle,i_obstacle = np.where(distance_from_center<obstacle_radius)
    pressureCells_Mask[j_obstacle,i_obstacle] = True
    
    # number of actual pressure cells
    #Test case scenario, when q=1 uniformly
#    Np = len(np.where(pressureCells_Mask==False)[0]) #Best for storing uniform conditions in q
    #Generate 1D arrays jj_C and ii_C containing all pressure cell indices where mask is false i.e. pressure is calculated
    jj_C,ii_C = np.where(pressureCells_Mask==False)
    #Assign q in a flattened (1D) array
    q  = 2*np.pi*n*np.sin(2*np.pi*n*Yc[jj_C,ii_C])*np.sin(2*np.pi*n*Xc[jj_C,ii_C])*((np.cos(2*np.pi*n*Xc[jj_C,ii_C])*np.sin(2*np.pi*n*Yc[jj_C,ii_C])) + (4*np.pi*n/Re))
    #Actual solution of the problem (for testing)
    sol = np.sin(2*np.pi*n*Xc[jj_C,ii_C])*np.sin(2*np.pi*n*Yc[jj_C,ii_C])
    #Test case scenario for q=1 uniformly
#    q = np.ones(Np)
    
    # a more advanced option is to separately create the divergence and gradient operators
    #Create laplacian using specified BCs
    DivGrad = spatial_operators.create_DivGrad_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask,BC)
    
    #Create del(phi)/del(x)
    Div = spatial_operators.create_delx_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask,sd,BC)
    # if boundary_conditions are not specified, it defaults to "Homogeneous Neumann"
    #keyboard()
    
    #Core Gauss-Seidel code block
    A = (DivGrad/Re)
    
    it = 1
    #Residual 
    var = 1
    phi_old = np.zeros(q.size)
    r_0 = np.linalg.norm(q - (phi_old*Div*phi_old) + (A*phi_old))
    residual = np.ones(1)

    #Specified tolerence for r_k/r_0
    print "Grid size = %d x %d, Re = %f omega = %2.1f" %(Nxc,Nyc,Re,omega)
    print "Wavenumber = %d, maximum number of iterations = %d" %(n,max_it)
    while var and it < max_it:
        Q = ((phi_old*Div*phi_old) - q)
        phi = spysparselinalg.spsolve(A,Q)
#        phi_star = np.dot(scysparse.linalg.inv(A1),Q)
        phi_old = phi
        r_k = np.linalg.norm(q + (A*phi) - (phi*Div*phi)) #Vector norm of error
#        print np.max(np.abs(phi-sol))
#        print it
#        print r_k
        print "Iteration: %d" %(it)
        print "Scaled residual: %2.14f" %(r_k/r_0)
        it += 1
        if (r_k/r_0) < tol:
            
#            print it            
            
            break
        
        residual = np.concatenate([residual,[r_k]])
        
        
            
        


    #Solution by direct solve    
    PHI = spysparselinalg.spsolve(A,q)
    
    #Pouring all 1D flattened array (of final solution using Gauss-Siedel, direct solve, and analytical evaluation of phi (sol))
    PHi = np.zeros((Nyc+2,Nxc+2))*np.nan
    PHi[np.where(pressureCells_Mask==False)] = PHI
    PHi_1 = np.zeros((Nyc+2,Nxc+2))*np.nan
    PHi_1[np.where(pressureCells_Mask==False)] = phi
    SOL = np.zeros((Nyc+2,Nxc+2))*np.nan
    SOL[np.where(pressureCells_Mask==False)] = sol
    
    #Plotter code for the contours of all solutions for comparison
#    figwidth       = 10
#    figheight      = 6
#    lineWidth      = 4
#    textFontSize   = 28
#    gcafontSize    = 30
#    
##    figure_name_1 = "Residual history for n=(%d) and w=(%2.3f).pdf" (%n) (%omega)  
#    
#    fig1 = plt.figure(0, figsize=(figwidth,figheight))
#    plt.plot(iteration[1:],residual[1:])
#    plt.setp(plt.xticklabels(),fontsize=gcafontSize)
#    plt.setp(plt.yticklabels(),fontsize=gcafontSize)
#    plt.grid('on',which='both')
##    plt.set_xlabel(r"iteration no. k",fontsize=textFontSize)
##    plt.set_ylabel(r"$r_{k}$",fontsize=textFontSize,rotation=90)
#    plt.title("Residual history for n= (%d), w = (%2.3f)" %n %omega)
#    plt.legend(loc='best')
#    plt.show()
    
#    # Plot solution. For code validation
#    fig = plt.figure(0, figsize=(figwidth,figheight))
#    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#    plt.axes(ax)
#    plt.contourf(Xc,Yc,PHi)
#    plt.colorbar()
#    
#    fig = plt.figure(1, figsize=(figwidth,figheight))
#    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#    plt.axes(ax)
#    plt.contourf(Xc,Yc,SOL)
#    plt.colorbar()
#    
#    fig = plt.figure(2, figsize=(figwidth,figheight))
#    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#    plt.axes(ax)
#    plt.contourf(Xc,Yc,PHi_1)
#    plt.colorbar()
##    
#    ax.grid('on',which='both')
#    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
#    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
#    ax.set_xlabel(r"$x$",fontsize=textFontSize)
#    ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
#    plt.axis("tight")
#    plt.axis("equal")
#    
#    plt.show()
#    
#    fig = plt.figure(1, figsize=(figwidth,figheight))
#    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#    plt.axes(ax)
#    plt.contourf(Xc,Yc,PHi_1)
#    plt.colorbar()
#    
#    ax.grid('on',which='both')
#    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
#    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
#    ax.set_xlabel(r"$x$",fontsize=textFontSize)
#    ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
#    plt.axis("tight")
#    plt.axis("equal")
#    
#    plt.show()
#    
#    fig = plt.figure(2, figsize=(figwidth,figheight))
#    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#    plt.axes(ax)
#    plt.contourf(Xc,Yc,SOL)
#    plt.colorbar()
#    
#    ax.grid('on',which='both')
#    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
#    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
#    ax.set_xlabel(r"$x$",fontsize=textFontSize)
#    ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
#    plt.axis("tight")
#    plt.axis("equal")
#    
#    plt.show()
#        
#    
#    
#    
##    Figure settings
#    matplotlibrc('text.latex', preamble='\usepackage{color}')
#    matplotlibrc('text',usetex=True)
#    matplotlibrc('font', family='serif')
    
    return it
    
#With pre-factorization. the splu generates an splu object which needs LU.solve(RHS)
def NLa_1(Nxc,Nyc,Lx,Ly,Re,n,omega,BC,it_max,tol,sd):
    # number of (pressure) cells = mass conservation cells
    #########################################
    ######## Preprocessing Stage ############
    
    # You might have to include ghost-cells here
    # Depending on your application
    
    # define grid for u and v velocity components first
    # and then define pressure cells locations
    xsi_u = np.linspace(0.,1.0,Nxc+1)
    xsi_v = np.linspace(0.,1.0,Nyc+1)
    # uniform grid
    xu = xsi_u*Lx
    yv = xsi_v*Ly
    # non-uniform grid can be specified by redefining xsi_u and xsi_v. For e.g.
    #xu = (xsi_u**4.)*Lx # can be replaced by non-uniform grid
    #yv = (xsi_v**4.)*Ly # can be replaced by non-uniform grid
    
    # creating ghost cells
    #internodal distance at the beginning and end
    dxu0 = np.diff(xu)[0]
    dxuL = np.diff(xu)[-1]
    #Concatenate the ghost points (obtained by flipping the same distances on either side of the first and last point)
    xu = np.concatenate([[xu[0]-dxu0],xu,[xu[-1]+dxuL]])
    
    dyv0 = np.diff(yv)[0]
    dyvL = np.diff(yv)[-1]
    yv = np.concatenate([[yv[0]-dyv0],yv,[yv[-1]+dyvL]])
    
    
    #Pressure nodes are cell centres
    #Velocity nodes described above are staggered wrt pressure nodes.
    dxc = np.diff(xu)  # pressure-cells spacings
    dyc = np.diff(yv)
    
    xc = 0.5*(xu[:-1]+xu[1:])  # total of Nxc cells
    yc = 0.5*(yv[:-1]+yv[1:])  # total of Nyc cells
    
    # note that indexing is Xc[j_y,i_x] or Xc[j,i]
    [Xc,Yc]   = np.meshgrid(xc,yc)     # these arrays are with ghost cells
    [Dxc,Dyc] = np.meshgrid(dxc,dyc)   # these arrays are with ghost cells
    
    ### familiarize yourself with 'flattening' options
    # phi_Fordering     = Phi.flatten(order='F') # F (column-major), Fortran/Matlab default
    # phi_Cordering     = Phi.flatten(order='C') # C (row-major), C/Python default
    # phi_PythonDefault = Phi.flatten()          # Python default
    # compare Phi[:,0] and Phi[0,:] with phi[:Nxc]
    
    #Mask has been flipped as compared to the starter code.
    #Mask is designed to be false for all points where pressure is to be calculated i.e. Mask==True means pressure is masked at those regions
    #This required changing np.zeros to np.ones in line 71
    # Pre-allocated False = fluid points
    pressureCells_Mask = np.ones(Xc.shape)
    pressureCells_Mask[1:-1,1:-1] = False # note that ghost cells are also marked true as a default value of 1 is assigned
    
    # Introducing obstacle in pressure Mask
    obstacle_radius = 0.0*Lx # set equal to 0.0*Lx to remove obstacle
    distance_from_center = np.sqrt(np.power(Xc-Lx/2.,2.0)+np.power(Yc-Ly/2.,2.0))
    j_obstacle,i_obstacle = np.where(distance_from_center<obstacle_radius)
    pressureCells_Mask[j_obstacle,i_obstacle] = True
    
    # number of actual pressure cells
    #Test case scenario, when q=1 uniformly
#    Np = len(np.where(pressureCells_Mask==False)[0]) #Best for storing uniform conditions in q
    #Generate 1D arrays jj_C and ii_C containing all pressure cell indices where mask is false i.e. pressure is calculated
    jj_C,ii_C = np.where(pressureCells_Mask==False)
    #Assign q in a flattened (1D) array
    q  = 2*np.pi*n*np.sin(2*np.pi*n*Yc[jj_C,ii_C])*np.sin(2*np.pi*n*Xc[jj_C,ii_C])*((np.cos(2*np.pi*n*Xc[jj_C,ii_C])*np.sin(2*np.pi*n*Yc[jj_C,ii_C])) + (4*np.pi*n/Re))
    #Actual solution of the problem (for testing)
    sol = np.sin(2*np.pi*n*Xc[jj_C,ii_C])*np.sin(2*np.pi*n*Yc[jj_C,ii_C])
    #Test case scenario for q=1 uniformly
#    q = np.ones(Np)
    
    # a more advanced option is to separately create the divergence and gradient operators
    #Create laplacian using specified BCs
    DivGrad = spatial_operators.create_DivGrad_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask,BC)
    
    #Create del(phi)/del(x)
    Div = spatial_operators.create_delx_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask,sd,BC)
    # if boundary_conditions are not specified, it defaults to "Homogeneous Neumann"
    #keyboard()
    
    #Core Gauss-Seidel code block
    A = (DivGrad/Re)
    
    it = 1
    #Residual 
    var = 1
    phi_old = np.zeros(q.size)
    r_0 = np.linalg.norm(q - (phi_old*Div*phi_old) + (A*phi_old))
    residual = np.ones(1)

    #Specified tolerence for r_k/r_0
    print "Grid size = %d x %d, Re = %f omega = %2.1f" %(Nxc,Nyc,Re,omega)
    print "Wavenumber = %d, maximum number of iterations = %d" %(n,max_it)
    while var and it < max_it:
        Q = ((phi_old*Div*phi_old) - q)
	LU = spysparselinalg.splu(A.tocsc())  #better to give it tocsc
        #phi = spysparselinalg.spsolve(LU.tocsr(),Q)
	phi = LU.solve(Q)
#        phi_star = np.dot(scysparse.linalg.inv(A1),Q)
        phi_old = phi
        r_k = np.linalg.norm(q + (A*phi) - (phi*Div*phi)) #Vector norm of error
#        print np.max(np.abs(phi-sol))
#        print it
#        print r_k
        print "Iteration: %d" %(it)
        print "Scaled residual: %2.14f" %(r_k/r_0)
        it += 1
        if (r_k/r_0) < tol:
            
#            print it            
            
            break
        
        residual = np.concatenate([residual,[r_k]])
        
        
            
        


    #Solution by direct solve    
    PHI = spysparselinalg.spsolve(A,q)
    
    #Pouring all 1D flattened array (of final solution using Gauss-Siedel, direct solve, and analytical evaluation of phi (sol))
    PHi = np.zeros((Nyc+2,Nxc+2))*np.nan
    PHi[np.where(pressureCells_Mask==False)] = PHI
    PHi_1 = np.zeros((Nyc+2,Nxc+2))*np.nan
    PHi_1[np.where(pressureCells_Mask==False)] = phi
    SOL = np.zeros((Nyc+2,Nxc+2))*np.nan
    SOL[np.where(pressureCells_Mask==False)] = sol
    
    #Plotter code for the contours of all solutions for comparison
#    figwidth       = 10
#    figheight      = 6
#    lineWidth      = 4
#    textFontSize   = 28
#    gcafontSize    = 30
#    
##    figure_name_1 = "Residual history for n=(%d) and w=(%2.3f).pdf" (%n) (%omega)  
#    
#    fig1 = plt.figure(0, figsize=(figwidth,figheight))
#    plt.plot(iteration[1:],residual[1:])
#    plt.setp(plt.xticklabels(),fontsize=gcafontSize)
#    plt.setp(plt.yticklabels(),fontsize=gcafontSize)
#    plt.grid('on',which='both')
##    plt.set_xlabel(r"iteration no. k",fontsize=textFontSize)
##    plt.set_ylabel(r"$r_{k}$",fontsize=textFontSize,rotation=90)
#    plt.title("Residual history for n= (%d), w = (%2.3f)" %n %omega)
#    plt.legend(loc='best')
#    plt.show()
    
#    # Plot solution. For code validation
#    fig = plt.figure(0, figsize=(figwidth,figheight))
#    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#    plt.axes(ax)
#    plt.contourf(Xc,Yc,PHi)
#    plt.colorbar()
#    
#    fig = plt.figure(1, figsize=(figwidth,figheight))
#    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#    plt.axes(ax)
#    plt.contourf(Xc,Yc,SOL)
#    plt.colorbar()
#    
#    fig = plt.figure(2, figsize=(figwidth,figheight))
#    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#    plt.axes(ax)
#    plt.contourf(Xc,Yc,PHi_1)
#    plt.colorbar()
##    
#    ax.grid('on',which='both')
#    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
#    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
#    ax.set_xlabel(r"$x$",fontsize=textFontSize)
#    ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
#    plt.axis("tight")
#    plt.axis("equal")
#    
#    plt.show()
#    
#    fig = plt.figure(1, figsize=(figwidth,figheight))
#    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#    plt.axes(ax)
#    plt.contourf(Xc,Yc,PHi_1)
#    plt.colorbar()
#    
#    ax.grid('on',which='both')
#    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
#    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
#    ax.set_xlabel(r"$x$",fontsize=textFontSize)
#    ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
#    plt.axis("tight")
#    plt.axis("equal")
#    
#    plt.show()
#    
#    fig = plt.figure(2, figsize=(figwidth,figheight))
#    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#    plt.axes(ax)
#    plt.contourf(Xc,Yc,SOL)
#    plt.colorbar()
#    
#    ax.grid('on',which='both')
#    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
#    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
#    ax.set_xlabel(r"$x$",fontsize=textFontSize)
#    ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
#    plt.axis("tight")
#    plt.axis("equal")
#    
#    plt.show()
#        
#    
#    
#    
##    Figure settings
#    matplotlibrc('text.latex', preamble='\usepackage{color}')
#    matplotlibrc('text',usetex=True)
#    matplotlibrc('font', family='serif')
    
    return it
      
def NLb(Nxc,Nyc,Lx,Ly,Re,n,omega,BC,it_max,tol,sd):
    # number of (pressure) cells = mass conservation cells
    #########################################
    ######## Preprocessing Stage ############
    
    # You might have to include ghost-cells here
    # Depending on your application
    
    # define grid for u and v velocity components first
    # and then define pressure cells locations
    xsi_u = np.linspace(0.,1.0,Nxc+1)
    xsi_v = np.linspace(0.,1.0,Nyc+1)
    # uniform grid
    xu = xsi_u*Lx
    yv = xsi_v*Ly
    # non-uniform grid can be specified by redefining xsi_u and xsi_v. For e.g.
    #xu = (xsi_u**4.)*Lx # can be replaced by non-uniform grid
    #yv = (xsi_v**4.)*Ly # can be replaced by non-uniform grid
    
    # creating ghost cells
    #internodal distance at the beginning and end
    dxu0 = np.diff(xu)[0]
    dxuL = np.diff(xu)[-1]
    #Concatenate the ghost points (obtained by flipping the same distances on either side of the first and last point)
    xu = np.concatenate([[xu[0]-dxu0],xu,[xu[-1]+dxuL]])
    
    dyv0 = np.diff(yv)[0]
    dyvL = np.diff(yv)[-1]
    yv = np.concatenate([[yv[0]-dyv0],yv,[yv[-1]+dyvL]])
    
    
    #Pressure nodes are cell centres
    #Velocity nodes described above are staggered wrt pressure nodes.
    dxc = np.diff(xu)  # pressure-cells spacings
    dyc = np.diff(yv)
    
    xc = 0.5*(xu[:-1]+xu[1:])  # total of Nxc cells
    yc = 0.5*(yv[:-1]+yv[1:])  # total of Nyc cells
    
    # note that indexing is Xc[j_y,i_x] or Xc[j,i]
    [Xc,Yc]   = np.meshgrid(xc,yc)     # these arrays are with ghost cells
    [Dxc,Dyc] = np.meshgrid(dxc,dyc)   # these arrays are with ghost cells
    
    ### familiarize yourself with 'flattening' options
    # phi_Fordering     = Phi.flatten(order='F') # F (column-major), Fortran/Matlab default
    # phi_Cordering     = Phi.flatten(order='C') # C (row-major), C/Python default
    # phi_PythonDefault = Phi.flatten()          # Python default
    # compare Phi[:,0] and Phi[0,:] with phi[:Nxc]
    
    #Mask has been flipped as compared to the starter code.
    #Mask is designed to be false for all points where pressure is to be calculated i.e. Mask==True means pressure is masked at those regions
    #This required changing np.zeros to np.ones in line 71
    # Pre-allocated False = fluid points
    pressureCells_Mask = np.ones(Xc.shape)
    pressureCells_Mask[1:-1,1:-1] = False # note that ghost cells are also marked true as a default value of 1 is assigned
    
    # Introducing obstacle in pressure Mask
    obstacle_radius = 0.0*Lx # set equal to 0.0*Lx to remove obstacle
    distance_from_center = np.sqrt(np.power(Xc-Lx/2.,2.0)+np.power(Yc-Ly/2.,2.0))
    j_obstacle,i_obstacle = np.where(distance_from_center<obstacle_radius)
    pressureCells_Mask[j_obstacle,i_obstacle] = True
    
    # number of actual pressure cells
    #Test case scenario, when q=1 uniformly
#    Np = len(np.where(pressureCells_Mask==False)[0]) #Best for storing uniform conditions in q
    #Generate 1D arrays jj_C and ii_C containing all pressure cell indices where mask is false i.e. pressure is calculated
    jj_C,ii_C = np.where(pressureCells_Mask==False)
    #Assign q in a flattened (1D) array
    q  = 2*np.pi*n*np.sin(2*np.pi*n*Yc[jj_C,ii_C])*np.sin(2*np.pi*n*Xc[jj_C,ii_C])*((np.cos(2*np.pi*n*Xc[jj_C,ii_C])*np.sin(2*np.pi*n*Yc[jj_C,ii_C])) + (4*np.pi*n/Re))
    #Actual solution of the problem (for testing)
    sol = np.sin(2*np.pi*n*Xc[jj_C,ii_C])*np.sin(2*np.pi*n*Yc[jj_C,ii_C])
    #Test case scenario for q=1 uniformly
#    q = np.ones(Np)
    
    # a more advanced option is to separately create the divergence and gradient operators
    #Create laplacian using specified BCs
    DivGrad = spatial_operators.create_DivGrad_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask,BC)
    
    #Create del(phi)/del(x)
    Div = spatial_operators.create_delx_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask,sd,BC)
    # if boundary_conditions are not specified, it defaults to "Homogeneous Neumann"
    #keyboard()
    
    #Core Gauss-Seidel code block
    A = (DivGrad/Re)
    
    it = 1
    #Residual 
    var = 1
    phi_old = 0.5*np.ones(q.size)
    r_0 = np.linalg.norm(q - (phi_old*Div*phi_old) + (A*phi_old))
    residual = np.ones(1)

    #Specified tolerence for r_k/r_0
    print "Grid size = %d x %d, Re = %f omega = %2.1f" %(Nxc,Nyc,Re,omega)
    print "Wavenumber = %d, maximum number of iterations = %d" %(n,max_it)
    while var and it < max_it:
        r_sit = np.ones(q.shape)
        sit = 1
        Q = ((A*phi_old) + q)
        phi_star_old = phi_old
        while var and sit < max_it:
            Q1 = Q/phi_star_old
            phi = spysparselinalg.spsolve(Div,Q1)
            r_sit = np.linalg.norm(phi-phi_star_old)
            print "            update = %f" %(np.linalg.norm(phi))
            print "            update_old = %f" %(np.linalg.norm(phi_star_old))
            phi_star_old = phi
            print "            Sub-Iteration: %d" %(sit)
            print "            Scaled residual for sub-iteration: %2.14f" %(r_sit)
            sit += 1
            if r_sit < 10:
                break
#        phi_star = np.dot(scysparse.linalg.inv(A1),Q)
        
        phi_old = phi
        r_k = np.linalg.norm(q + (A*phi) - (phi*Div*phi)) #Vector norm of error
#        print np.max(np.abs(phi-sol))
#        print it
#        print r_k
        print "Iteration: %d" %(it)
        print "Scaled residual: %2.14f" %(r_k/r_0)
        
        if (r_k/r_0) < tol:
            
#            print it            
            
            break
        it += 1
        residual = np.concatenate([residual,[r_k]])
        
        
            
        


    #Solution by direct solve    
    PHI = spysparselinalg.spsolve(A,q)
    
    #Pouring all 1D flattened array (of final solution using Gauss-Siedel, direct solve, and analytical evaluation of phi (sol))
    PHi = np.zeros((Nyc+2,Nxc+2))*np.nan
    PHi[np.where(pressureCells_Mask==False)] = PHI
    PHi_1 = np.zeros((Nyc+2,Nxc+2))*np.nan
    PHi_1[np.where(pressureCells_Mask==False)] = phi
    SOL = np.zeros((Nyc+2,Nxc+2))*np.nan
    SOL[np.where(pressureCells_Mask==False)] = sol
    
    #Plotter code for the contours of all solutions for comparison
#    figwidth       = 10
#    figheight      = 6
#    lineWidth      = 4
#    textFontSize   = 28
#    gcafontSize    = 30
#    
##    figure_name_1 = "Residual history for n=(%d) and w=(%2.3f).pdf" (%n) (%omega)  
#    
#    fig1 = plt.figure(0, figsize=(figwidth,figheight))
#    plt.plot(iteration[1:],residual[1:])
#    plt.setp(plt.xticklabels(),fontsize=gcafontSize)
#    plt.setp(plt.yticklabels(),fontsize=gcafontSize)
#    plt.grid('on',which='both')
##    plt.set_xlabel(r"iteration no. k",fontsize=textFontSize)
##    plt.set_ylabel(r"$r_{k}$",fontsize=textFontSize,rotation=90)
#    plt.title("Residual history for n= (%d), w = (%2.3f)" %n %omega)
#    plt.legend(loc='best')
#    plt.show()
    
#    # Plot solution. For code validation
#    fig = plt.figure(0, figsize=(figwidth,figheight))
#    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#    plt.axes(ax)
#    plt.contourf(Xc,Yc,PHi)
#    plt.colorbar()
#    
#    fig = plt.figure(1, figsize=(figwidth,figheight))
#    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#    plt.axes(ax)
#    plt.contourf(Xc,Yc,SOL)
#    plt.colorbar()
#    
#    fig = plt.figure(2, figsize=(figwidth,figheight))
#    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#    plt.axes(ax)
#    plt.contourf(Xc,Yc,PHi_1)
#    plt.colorbar()
##    
#    ax.grid('on',which='both')
#    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
#    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
#    ax.set_xlabel(r"$x$",fontsize=textFontSize)
#    ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
#    plt.axis("tight")
#    plt.axis("equal")
#    
#    plt.show()
#    
#    fig = plt.figure(1, figsize=(figwidth,figheight))
#    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#    plt.axes(ax)
#    plt.contourf(Xc,Yc,PHi_1)
#    plt.colorbar()
#    
#    ax.grid('on',which='both')
#    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
#    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
#    ax.set_xlabel(r"$x$",fontsize=textFontSize)
#    ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
#    plt.axis("tight")
#    plt.axis("equal")
#    
#    plt.show()
#    
#    fig = plt.figure(2, figsize=(figwidth,figheight))
#    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#    plt.axes(ax)
#    plt.contourf(Xc,Yc,SOL)
#    plt.colorbar()
#    
#    ax.grid('on',which='both')
#    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
#    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
#    ax.set_xlabel(r"$x$",fontsize=textFontSize)
#    ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
#    plt.axis("tight")
#    plt.axis("equal")
#    
#    plt.show()
#        
#    
#    
#    
##    Figure settings
#    matplotlibrc('text.latex', preamble='\usepackage{color}')
#    matplotlibrc('text',usetex=True)
#    matplotlibrc('font', family='serif')
    
    return it

def U(t,X,Y):
    u = -np.cos(X[1:-1,1:-1])*np.sin(Y[1:-1,1:-1])
    return u
    
def V(t,X,Y):
    v = np.sin(X[1:-1,1:-1])*np.cos(Y[1:-1,1:-1])
    return v
    
def P(t,X,Y):
    p = -0.25*(np.cos(2*X[1:-1,1:-1])+np.cos(2*Y[1:-1,1:-1]))
    return p
    

#########################################
############### Code Starts ##############
figure_folder = "../report/"
Problem_1_spy = True
Problem_1_parta = True
Problem_1_partb = True
Problem_1_partb_1st = True
Problem_1_partc = True
Problem_2 = True

if Problem_1_spy:
    
    Nxc = 5
    Nyc = 5
    Lx = 1.0
    Ly = 1.0
    # number of (pressure) cells = mass conservation cells
    #########################################
    ######## Preprocessing Stage ############
    
    # You might have to include ghost-cells here
    # Depending on your application
    
    # define grid for u and v velocity components first
    # and then define pressure cells locations
    xsi_u = np.linspace(0.,1.0,Nxc+1)
    xsi_v = np.linspace(0.,1.0,Nyc+1)
    # uniform grid
    xu = xsi_u*Lx
    yv = xsi_v*Ly
    # non-uniform grid can be specified by redefining xsi_u and xsi_v. For e.g.
    #xu = (xsi_u**4.)*Lx # can be replaced by non-uniform grid
    #yv = (xsi_v**4.)*Ly # can be replaced by non-uniform grid
    
    # creating ghost cells
    #internodal distance at the beginning and end
    dxu0 = np.diff(xu)[0]
    dxuL = np.diff(xu)[-1]
    #Concatenate the ghost points (obtained by flipping the same distances on either side of the first and last point)
    xu = np.concatenate([[xu[0]-dxu0],xu,[xu[-1]+dxuL]])
    
    dyv0 = np.diff(yv)[0]
    dyvL = np.diff(yv)[-1]
    yv = np.concatenate([[yv[0]-dyv0],yv,[yv[-1]+dyvL]])
    
    
    #Pressure nodes are cell centres
    #Velocity nodes described above are staggered wrt pressure nodes.
    dxc = np.diff(xu)  # pressure-cells spacings
    dyc = np.diff(yv)
    
    xc = 0.5*(xu[:-1]+xu[1:])  # total of Nxc cells
    yc = 0.5*(yv[:-1]+yv[1:])  # total of Nyc cells
    
    # note that indexing is Xc[j_y,i_x] or Xc[j,i]
    [Xc,Yc]   = np.meshgrid(xc,yc)     # these arrays are with ghost cells
    [Dxc,Dyc] = np.meshgrid(dxc,dyc)   # these arrays are with ghost cells
    
    ### familiarize yourself with 'flattening' options
    # phi_Fordering     = Phi.flatten(order='F') # F (column-major), Fortran/Matlab default
    # phi_Cordering     = Phi.flatten(order='C') # C (row-major), C/Python default
    # phi_PythonDefault = Phi.flatten()          # Python default
    # compare Phi[:,0] and Phi[0,:] with phi[:Nxc]
    
    #Mask has been flipped as compared to the starter code.
    #Mask is designed to be false for all points where pressure is to be calculated i.e. Mask==True means pressure is masked at those regions
    #This required changing np.zeros to np.ones in line 71
    # Pre-allocated False = fluid points
    pressureCells_Mask = np.ones(Xc.shape)
    pressureCells_Mask[1:-1,1:-1] = False # note that ghost cells are also marked true as a default value of 1 is assigned
    
    # Introducing obstacle in pressure Mask
    obstacle_radius = 0.0*Lx # set equal to 0.0*Lx to remove obstacle
    distance_from_center = np.sqrt(np.power(Xc-Lx/2.,2.0)+np.power(Yc-Ly/2.,2.0))
    j_obstacle,i_obstacle = np.where(distance_from_center<obstacle_radius)
    pressureCells_Mask[j_obstacle,i_obstacle] = True
    
    # a more advanced option is to separately create the divergence and gradient operators
    #Create laplacian using specified BCs
    BC = "Homogeneous Dirichlet"
    DivGrad = spatial_operators.create_DivGrad_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask,BC)
    figwidth       = 10
    figheight      = 6
    lineWidth      = 4
    textFontSize   = 14
    gcafontSize    = 30
    figure_name = "Spy plot for Homogeneous Dirichlet.pdf"
    figure_file = figure_folder + figure_name       
    fig = plt.spy(DivGrad)
    print "Saving Figure:" + figure_file
    plt.savefig(figure_file)
    
    BC = "Homogeneous Neumann"
    DivGrad = spatial_operators.create_DivGrad_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask,BC)
    figwidth       = 10
    figheight      = 6
    lineWidth      = 4
    textFontSize   = 14
    gcafontSize    = 30
    figure_name = "Spy plot for Homogeneous Neumann.pdf"
    figure_file = figure_folder + figure_name       
    fig = plt.spy(DivGrad)
    print "Saving Figure:" + figure_file
    plt.savefig(figure_file)


if Problem_1_parta:
    
    #Part a
    
    figwidth       = 10
    figheight      = 6
    lineWidth      = 4
    textFontSize   = 14
    gcafontSize    = 30
    tol = 10**-(10)
    Nxc  = 15
    Nyc  = 15
    Lx   = 1.
    Ly   = 1.
    Re = 1
    omega = np.linspace(1.0,1.81)
    max_it = 10000
    N = np.array([1,np.min(np.array([Nxc,Nyc]))/2,np.min(np.array([Nxc,Nyc]))])
    opt_omega = np.zeros(N.size)
    opt_it_no = np.zeros(N.size)
    iteration_number = np.zeros((N.size,omega.size))
    residual_opt = np.zeros((N.size,max_it+1))
    for i,n in enumerate(N):
        for j,o in enumerate(omega):
            iteration_number[i,j],residual = Gauss_Seidel(Nxc,Nyc,Lx,Ly,Re,n,o,"Homogeneous Dirichlet",max_it,tol,"2nd-order-central")
            if iteration_number[i,j] > iteration_number[i,j-1] and j>0:
                break
            residual_opt[i,0:iteration_number[i,j]] = residual
            opt_omega[i] = o
            opt_it_no[i] = iteration_number[i,j]
            
    
    figure_name = "Problem1_Part_a_1_%d x %d.pdf" %(Nxc,Nyc)
    figure_file = figure_folder + figure_name       
    fig = plt.figure(figsize=(figwidth,figheight))
    plt.grid('on',which='both')
    plt.xlabel("$\omega$",fontsize=textFontSize)
    plt.ylabel(r"No. of iterations",fontsize=textFontSize,rotation=90)
    plt.title("No. of iterations vs $\omega$")
    plt.semilogy(omega,iteration_number[0,:],label="n=%d" %(N[0]))
    plt.semilogy(omega,iteration_number[1,:],label="n=%d" %(N[1]))
    plt.semilogy(omega,iteration_number[2,:],label="n=%d" %(N[2]))
    plt.legend(loc='best')
    
    print "Saving Figure:" + figure_file
    plt.savefig(figure_file)
    
    figure_name = "Problem1_Part_a_2_%d x %d.pdf" %(Nxc,Nyc)
    figure_file = figure_folder + figure_name       
    fig = plt.figure(figsize=(figwidth,figheight))
    plt.grid('on',which='both')
    plt.xlabel("No. of iterations",fontsize=textFontSize)
    plt.ylabel(r"$||r_k||$",fontsize=textFontSize,rotation=90)
    plt.title("$||r_k||$ vs No. of iterations for optimum $\omega$")
    plt.loglog(residual_opt[0,0:opt_it_no[0]],label="n=%d,$\omega_{opt}$=%f" %(N[0],opt_omega[0]))
    plt.loglog(residual_opt[1,0:opt_it_no[1]],label="n=%d,$\omega_{opt}$=%f" %(N[1],opt_omega[1]))
    plt.loglog(residual_opt[2,0:opt_it_no[2]],label="n=%d,$\omega_{opt}$=%f" %(N[2],opt_omega[2]))
    plt.legend(loc='best')
    
    print "Saving Figure:" + figure_file
    plt.savefig(figure_file)
    
    Nxc  = 30
    Nyc  = 30
    Lx   = 1.
    Ly   = 1.
    Re = 1
    omega = np.linspace(1.0,1.8,11)
    N = np.array([1,np.min(np.array([Nxc,Nyc]))/2,np.min(np.array([Nxc,Nyc]))])
    opt_omega = np.zeros(N.size)
    opt_it_no = np.zeros(N.size)
    iteration_number = np.zeros((N.size,omega.size))
    residual_opt = np.zeros((N.size,max_it+1))
    for i,n in enumerate(N):
        for j,o in enumerate(omega):
            iteration_number[i,j],residual = Gauss_Seidel(Nxc,Nyc,Lx,Ly,Re,n,o,"Homogeneous Dirichlet",max_it,tol,"2nd-order-central")
            if iteration_number[i,j] > iteration_number[i,j-1] and j>0:
                break
            residual_opt[i,0:iteration_number[i,j]] = residual
            opt_omega[i] = o
            opt_it_no[i] = iteration_number[i,j]
            
    
    figure_name = "Problem1_Part_a_1_%d x %d.pdf" %(Nxc,Nyc)
    figure_file = figure_folder + figure_name       
    fig = plt.figure(figsize=(figwidth,figheight))
    plt.grid('on',which='both')
    plt.xlabel("$\omega$",fontsize=textFontSize)
    plt.ylabel(r"Iterations",fontsize=textFontSize,rotation=90)
    plt.title("No. of iterations vs $\omega$")
    plt.semilogy(omega,iteration_number[0,:],label="n=%d" %(N[0]))
    plt.semilogy(omega,iteration_number[1,:],label="n=%d" %(N[1]))
    plt.semilogy(omega,iteration_number[2,:],label="n=%d" %(N[2]))
    plt.legend(loc='best')
    
    print "Saving Figure:" + figure_file
    plt.savefig(figure_file)
    
    figure_name = "Problem1_Part_a_2_%d x %d.pdf" %(Nxc,Nyc)
    figure_file = figure_folder + figure_name       
    fig = plt.figure(figsize=(figwidth,figheight))
    plt.grid('on',which='both')
    plt.xlabel("No. of iterations",fontsize=textFontSize)
    plt.ylabel(r"$||r_k||$",fontsize=textFontSize,rotation=90)
    plt.title("$||r_k||$ vs No. of iterations for optimum $\omega$")
    plt.loglog(residual_opt[0,0:opt_it_no[0]],label="n=%d,$\omega_{opt}$=%f" %(N[0],opt_omega[0]))
    plt.loglog(residual_opt[1,0:opt_it_no[1]],label="n=%d,$\omega_{opt}$=%f" %(N[1],opt_omega[1]))
    plt.loglog(residual_opt[2,0:opt_it_no[2]],label="n=%d,$\omega_{opt}$=%f" %(N[2],opt_omega[2]))
    plt.legend(loc='best')
    
    print "Saving Figure:" + figure_file
    plt.savefig(figure_file)
    
    Nxc  = 45
    Nyc  = 45
    Lx   = 1.
    Ly   = 1.
    Re = 1
    omega = np.linspace(1.0,1.8,11)
    N = np.array([1,np.min(np.array([Nxc,Nyc]))/2,np.min(np.array([Nxc,Nyc]))])
    opt_omega = np.zeros(N.size)
    opt_it_no = np.zeros(N.size)
    iteration_number = np.zeros((N.size,omega.size))
    residual_opt = np.zeros((N.size,max_it+1))
    for i,n in enumerate(N):
        for j,o in enumerate(omega):
            iteration_number[i,j],residual = Gauss_Seidel(Nxc,Nyc,Lx,Ly,Re,n,o,"Homogeneous Dirichlet",max_it,tol,"2nd-order-central")
            if iteration_number[i,j] > iteration_number[i,j-1] and j>0:
                break
            residual_opt[i,0:iteration_number[i,j]] = residual
            opt_omega[i] = o
            opt_it_no[i] = iteration_number[i,j]
            
    
    figure_name = "Problem1_Part_a_1_%d x %d.pdf" %(Nxc,Nyc)
    figure_file = figure_folder + figure_name       
    fig = plt.figure(figsize=(figwidth,figheight))
    plt.grid('on',which='both')
    plt.xlabel("$\omega$",fontsize=textFontSize)
    plt.ylabel(r"Iterations",fontsize=textFontSize,rotation=90)
    plt.title("No. of iterations vs $\omega$")
    plt.semilogy(omega,iteration_number[0,:],label="n=%d" %(N[0]))
    plt.semilogy(omega,iteration_number[1,:],label="n=%d" %(N[1]))
    plt.semilogy(omega,iteration_number[2,:],label="n=%d" %(N[2]))
    plt.legend(loc='best')
    
    print "Saving Figure:" + figure_file
    plt.savefig(figure_file)
    
    figure_name = "Problem1_Part_a_2_%d x %d.pdf" %(Nxc,Nyc)
    figure_file = figure_folder + figure_name       
    fig = plt.figure(figsize=(figwidth,figheight))
    plt.grid('on',which='both')
    plt.xlabel("No. of iterations",fontsize=textFontSize)
    plt.ylabel(r"$||r_k||$",fontsize=textFontSize,rotation=90)
    plt.title("$||r_k||$ vs No. of iterations for optimum $\omega$")
    plt.loglog(residual_opt[0,0:opt_it_no[0]],label="n=%d,$\omega_{opt}$=%f" %(N[0],opt_omega[0]))
    plt.loglog(residual_opt[1,0:opt_it_no[1]],label="n=%d,$\omega_{opt}$=%f" %(N[1],opt_omega[1]))
    plt.loglog(residual_opt[2,0:opt_it_no[2]],label="n=%d,$\omega_{opt}$=%f" %(N[2],opt_omega[2]))
    plt.legend(loc='best')
    
    print "Saving Figure:" + figure_file
    plt.savefig(figure_file)

if Problem_1_partb:
    
    figwidth       = 10
    figheight      = 6
    lineWidth      = 4
    textFontSize   = 14
    gcafontSize    = 30
    Nxc  = 256
    Nyc  = 256
    Lx   = 1.
    Ly   = 1.
    Re = np.logspace(-2,8,6)
    omega = np.linspace(1.0,1.8,11)
    N = 128
    max_it = 500
    tol = 10**-10
    opt_omega = np.zeros(Re.size)
    opt_it_no = np.zeros(Re.size)
    iteration_number = np.zeros((Re.size,omega.size))
    residual_opt = np.zeros((Re.size,max_it+1))
    for i,re in enumerate(Re):
        for j,o in enumerate(omega):
            if re > 10 and o>1.05:
                break
            iteration_number[i,j],residual = Gauss_Seidel(Nxc,Nyc,Lx,Ly,re,N,o,"Homogeneous Dirichlet",max_it,tol,"1st-order-upwind")
            if iteration_number[i,j] > iteration_number[i,j-1] and j>0:
                print "Optimum omega = %f" %(opt_omega[i-1])
                print "No. of iterations for omega = %f: %d" %(opt_omega[i-1],opt_it_no[i-1])
                break
            residual_opt[i,0:iteration_number[i,j]] = residual
            opt_omega[i] = o
            opt_it_no[i] = iteration_number[i,j]
            
    
    figure_name = "Problem1_Part_b_%d x %d_with_1st-order-upwind.pdf" %(Nxc,Nyc)
    figure_file = figure_folder + figure_name       
    fig = plt.figure(figsize=(figwidth,figheight))
    plt.grid('on',which='both')
    plt.xlabel("No. of iterations",fontsize=textFontSize)
    plt.ylabel(r"$||r_k||$",fontsize=textFontSize,rotation=90)
    plt.title("$||r_k||$ vs No. of iterations for optimum $\omega$")
    plt.loglog(residual_opt[0,0:opt_it_no[0]],label="n=%2.2f,$\omega_{opt}$=%.2f" %(Re[0],opt_omega[0]))
    plt.loglog(residual_opt[1,0:opt_it_no[1]],label="n=%2.2f,$\omega_{opt}$=%.2f" %(Re[1],opt_omega[1]))
    plt.loglog(residual_opt[2,0:opt_it_no[2]],label="n=%2.2f,$\omega_{opt}$=%.2f" %(Re[2],opt_omega[2]))
    plt.loglog(residual_opt[3,0:opt_it_no[3]],label="n=%2.2f,$\omega_{opt}$=%.2f" %(Re[3],opt_omega[3]))
    plt.loglog(residual_opt[4,0:opt_it_no[4]],label="n=%2.2f,$\omega_{opt}$=%.2f" %(Re[4],opt_omega[4]))
    plt.loglog(residual_opt[5,0:opt_it_no[5]],label="n=%2.2f,$\omega_{opt}$=%.2f" %(Re[5],opt_omega[5]))
    plt.legend(loc='best')
    
    print "Saving Figure:" + figure_file
    plt.savefig(figure_file)
    
if Problem_1_partc:
    figwidth       = 10
    figheight      = 6
    lineWidth      = 4
    textFontSize   = 14
    gcafontSize    = 30
    Nxc  = 210
    Nyc  = 210
    Lx   = 1.
    Ly   = 1.
    Re = np.logspace(-2,8,6)
    omega = 1
    N = Nxc/2.0
    max_it = 10
    tol = 10**-6
    iteration_number1 = np.zeros(Re.size)
    iteration_number2 = np.zeros(Re.size)
    iteration_number3 = np.zeros(Re.size)
    start_1 = np.zeros(Re.size)
    end_1 = np.zeros(Re.size)
    start_2 = np.zeros(Re.size)
    end_2 = np.zeros(Re.size)
    start_3 = np.zeros(Re.size)
    end_3 = np.zeros(Re.size)
    for i,re in enumerate(Re):
	start_1[i] = time.time()
        iteration_number1[i] = NLa(Nxc,Nyc,Lx,Ly,re,N,omega,"Homogeneous Dirichlet",max_it,tol,"2nd-order-central")
	end_1[i] = time.time()
	start_2[i] = time.time()
        iteration_number2[i] = NLa_1(Nxc,Nyc,Lx,Ly,re,N,omega,"Homogeneous Dirichlet",max_it,tol,"2nd-order-central")
	end_2[i] = time.time()
	start_3[i] = time.time()
	iteration_number3[i] = NLb(Nxc,Nyc,Lx,Ly,re,N,omega,"Homogeneous Dirichlet",max_it,tol,"2nd-order-central")
	end_3[i] = time.time()
	

    figure_name = "Problem1_Part_c_1_%d x %d.pdf" %(Nxc,Nyc)
    figure_file = figure_folder + figure_name       
    fig = plt.figure(figsize=(figwidth,figheight))
    plt.grid('on',which='both')
    plt.xlabel("Re",fontsize=textFontSize)
    plt.ylabel(r"No. of iterations",fontsize=textFontSize,rotation=90)
    plt.title("No. of iterations vs Re")
    plt.loglog(Re,iteration_number1,'r--',label="Without Factorization")
    plt.loglog(Re,iteration_number2,'b--',label="With Factorization")
    plt.loglog(Re,iteration_number3,'g--',label="Method with sub-iterations")
    plt.legend(loc='best')
    print "Saving Figure:" + figure_file
    plt.savefig(figure_file)

    figure_name = "Problem1_Part_c_2_%d x %d.pdf" %(Nxc,Nyc)
    figure_file = figure_folder + figure_name       
    fig = plt.figure(figsize=(figwidth,figheight))
    plt.grid('on',which='both')
    plt.xlabel("Re",fontsize=textFontSize)
    plt.ylabel(r"Time Required",fontsize=textFontSize,rotation=90)
    plt.title("Time required vs Re")
    plt.loglog(Re,end_1-start_1,'r--',label="Without Factorization")
    plt.loglog(Re,end_2-start_2,'b--',label="With Factorization")
    plt.loglog(Re,end_3-start_3,'g--',label="Method with sub-iterations")
    plt.legend(loc='best')
    print "Saving Figure:" + figure_file
    plt.savefig(figure_file)
        
        
if Problem_2:
    figwidth       = 10
    figheight      = 6
    lineWidth      = 4
    textFontSize   = 14
    gcafontSize    = 30
    NXC  = np.array([8,40,128])
    NYC  = NXC
    Lx   = 2*np.pi
    Ly   = 2*np.pi
    nu = 1
    t_final = 1
    dt = 0.001
    
    
#    # number of (pressure) cells = mass conservation cells
#    #########################################
#    ######## Preprocessing Stage ############
#    
#    # You might have to include ghost-cells here
#    # Depending on your application
#    
#    # define grid for u and v velocity components first
#    # and then define pressure cells locations
#    xsi_u = np.linspace(0.,1.0,Nxc+1)
#    xsi_v = np.linspace(0.,1.0,Nyc+1)
#    # uniform grid
#    xu = xsi_u*Lx
#    yv = xsi_v*Ly
#    # non-uniform grid can be specified by redefining xsi_u and xsi_v. For e.g.
#    #xu = (xsi_u**4.)*Lx # can be replaced by non-uniform grid
#    #yv = (xsi_v**4.)*Ly # can be replaced by non-uniform grid
#    
#    # creating ghost cells
#    #internodal distance at the beginning and end
#    dxu0 = np.diff(xu)[0]
#    dxuL = np.diff(xu)[-1]
#    #Concatenate the ghost points (obtained by flipping the same distances on either side of the first and last point)
#    xu = np.concatenate([[xu[0]-dxu0],xu,[xu[-1]+dxuL]])
#    
#    dyv0 = np.diff(yv)[0]
#    dyvL = np.diff(yv)[-1]
#    yv = np.concatenate([[yv[0]-dyv0],yv,[yv[-1]+dyvL]])
#    
#    
#    #Pressure nodes are cell centres
#    #Velocity nodes described above are staggered wrt pressure nodes.
#    dxc = np.diff(xu)  # pressure-cells spacings
#    dyc = np.diff(yv)
#
#    
#    #Pressure points
#    xc = 0.5*(xu[:-1]+xu[1:])  # total of Nxc cells
#    yc = 0.5*(yv[:-1]+yv[1:])  # total of Nyc cells
#    
#    dxu = np.concatenate([[0],np.diff(xc),[0]])
#    dyv = np.concatenate([[0],np.diff(yc),[0]])
#    
#    # note that indexing is Xc[j_y,i_x] or Xc[j,i]
#    
#    #Pressure grid
#    [Xp,Yp]   = np.meshgrid(xc,yc)     # these arrays are with ghost cells
#    [Dxp,Dyp] = np.meshgrid(dxc,dyc)   # these arrays are with ghost cells
#    
#    #U grid
#    [Xu,Yu] = np.meshgrid(xu,yc)
#    [Dxu,Dyu] = np.meshgrid(dxu,dyc)
#    
#    #V grid
#    [Xv,Yv] = np.meshgrid(xc,yv)
#    [Dxv,Dyv] = np.meshgrid(dxc,dyv)
#    
#    ### familiarize yourself with 'flattening' options
#    # phi_Fordering     = Phi.flatten(order='F') # F (column-major), Fortran/Matlab default
#    # phi_Cordering     = Phi.flatten(order='C') # C (row-major), C/Python default
#    # phi_PythonDefault = Phi.flatten()          # Python default
#    # compare Phi[:,0] and Phi[0,:] with phi[:Nxc]
#    
#    #Mask has been flipped as compared to the starter code.
#    #Mask is designed to be false for all points where pressure is to be calculated i.e. Mask==True means pressure is masked at those regions
#    #This required changing np.zeros to np.ones in line 71
#    # Pre-allocated False = fluid points
#    pressureCells_Mask = np.ones(Xp.shape)
#    pressureCells_Mask[1:-1,1:-1] = False # note that ghost cells are also marked true as a default value of 1 is assigned
#    
#    u_velocityCells_Mask = np.ones(Xu.shape)
#    u_velocityCells_Mask[1:-1,1:-1] = False
#    v_velocityCells_Mask = np.ones(Xv.shape)
#    v_velocityCells_Mask [1:-1,1:-1] = False
#    # Introducing obstacle in pressure Mask
##    obstacle_radius = 0.0*Lx # set equal to 0.0*Lx to remove obstacle
##    distance_from_center = np.sqrt(np.power(Xp-Lx/2.,2.0)+np.power(Yp-Ly/2.,2.0))
##    j_obstacle,i_obstacle = np.where(distance_from_center<obstacle_radius)
##    pressureCells_Mask[j_obstacle,i_obstacle] = True
#    
#    sd = "2nd-order-central"
#    DivGradp = spatial_operators.create_DivGrad_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask)
#    DivGradu = spatial_operators.create_DivGrad_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
#    DivGradv = spatial_operators.create_DivGrad_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"periodic")
#    delxu = spatial_operators.create_delx_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,sd,"periodic")
#    delxp = spatial_operators.create_delx_operator(Dxu,Dyu,Xu,Yu,pressureCells_Mask,sd,"periodic")
#    delyu = spatial_operators.create_dely_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,sd,"periodic")
#    delxv = spatial_operators.create_delx_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,sd,"periodic")
#    delyv = spatial_operators.create_dely_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,sd,"periodic")
#    divxu = spatial_operators.create_divx_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,sd)
#    divyv = spatial_operators.create_divy_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,sd)
#    divxp = spatial_operators.create_divx_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,sd)
#    divyp = spatial_operators.create_divy_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,sd)
#
#    #initial values of u,v,p
    RK1 = True
    RK2 = True
    RK3 = True
    RK4 = True
#    u_initial = u(0,Xu,Yu)
#    v_initial = v(0,Xv,Yv)
#    p_initial = p(0,Xp,Yp)
#    u_final = u(1,Xu,Yu)
#    v_final = v(1,Xv,Yv)
#    p_final = p(1,Xp,Yp)
#    u = u_initial
#    v = v_initial
#    p = p_initial
#    u_star = np.zeros(u.size)
#    v_star = np.zeros(v.size)
#    nu = 1
#    t_final = 1
#    dt = 0.0001
#    time_t = 0
#    it = 1
#    
#    #Flattening u,v,p 2-D arrays
#    u_flattened = u.flatten()
#    v_flattened = v.flatten()
#    u_star_flattened = u_star.flatten()
#    v_star_flattened = v_star.flatten()
#    p_flattened = p.flatten()

    if RK1:
        e_TR_u_1 = np.zeros(NXC.size)
        e_TR_v_1 = np.zeros(NYC.size)
        for index,Nxc in enumerate(NXC):
            Nyc = int(NYC[index])
            Nxc = int(Nxc)
            print "Nxc = %d, Nyc = %d" %(Nxc,Nyc)
                    # number of (pressure) cells = mass conservation cells
            #########################################
            ######## Preprocessing Stage ############
            
            # You might have to include ghost-cells here
            # Depending on your application
            
            # define grid for u and v velocity components first
            # and then define pressure cells locations
            xsi_u = np.linspace(0.,1.0,Nxc+1)
            xsi_v = np.linspace(0.,1.0,Nyc+1)
            # uniform grid
            xu = xsi_u*Lx
            yv = xsi_v*Ly
            # non-uniform grid can be specified by redefining xsi_u and xsi_v. For e.g.
            #xu = (xsi_u**4.)*Lx # can be replaced by non-uniform grid
            #yv = (xsi_v**4.)*Ly # can be replaced by non-uniform grid
            
            # creating ghost cells
            #internodal distance at the beginning and end
            dxu0 = np.diff(xu)[0]
            dxuL = np.diff(xu)[-1]
            #Concatenate the ghost points (obtained by flipping the same distances on either side of the first and last point)
            xu = np.concatenate([[xu[0]-dxu0],xu,[xu[-1]+dxuL]])
            
            dyv0 = np.diff(yv)[0]
            dyvL = np.diff(yv)[-1]
            yv = np.concatenate([[yv[0]-dyv0],yv,[yv[-1]+dyvL]])
            
            
            #Pressure nodes are cell centres
            #Velocity nodes described above are staggered wrt pressure nodes.
            dxc = np.diff(xu)  # pressure-cells spacings
            dyc = np.diff(yv)
        
            
            #Pressure points
            xc = 0.5*(xu[:-1]+xu[1:])  # total of Nxc cells
            yc = 0.5*(yv[:-1]+yv[1:])  # total of Nyc cells
            
            dxu = np.concatenate([[0],np.diff(xc),[0]])
            dyv = np.concatenate([[0],np.diff(yc),[0]])
            
            # note that indexing is Xc[j_y,i_x] or Xc[j,i]
            
            #Pressure grid
            [Xp,Yp]   = np.meshgrid(xc,yc)     # these arrays are with ghost cells
            [Dxp,Dyp] = np.meshgrid(dxc,dyc)   # these arrays are with ghost cells
            
            #U grid
            [Xu,Yu] = np.meshgrid(xu,yc)
            [Dxu,Dyu] = np.meshgrid(dxu,dyc)
            
            #V grid
            [Xv,Yv] = np.meshgrid(xc,yv)
            [Dxv,Dyv] = np.meshgrid(dxc,dyv)
            
            ### familiarize yourself with 'flattening' options
            # phi_Fordering     = Phi.flatten(order='F') # F (column-major), Fortran/Matlab default
            # phi_Cordering     = Phi.flatten(order='C') # C (row-major), C/Python default
            # phi_PythonDefault = Phi.flatten()          # Python default
            # compare Phi[:,0] and Phi[0,:] with phi[:Nxc]
            
            #Mask has been flipped as compared to the starter code.
            #Mask is designed to be false for all points where pressure is to be calculated i.e. Mask==True means pressure is masked at those regions
            #This required changing np.zeros to np.ones in line 71
            # Pre-allocated False = fluid points
            pressureCells_Mask = np.ones(Xp.shape)
            pressureCells_Mask[1:-1,1:-1] = False # note that ghost cells are also marked true as a default value of 1 is assigned
            
            u_velocityCells_Mask = np.ones(Xu.shape)
            u_velocityCells_Mask[1:-1,1:-1] = False
            v_velocityCells_Mask = np.ones(Xv.shape)
            v_velocityCells_Mask [1:-1,1:-1] = False
            # Introducing obstacle in pressure Mask
        #    obstacle_radius = 0.0*Lx # set equal to 0.0*Lx to remove obstacle
        #    distance_from_center = np.sqrt(np.power(Xp-Lx/2.,2.0)+np.power(Yp-Ly/2.,2.0))
        #    j_obstacle,i_obstacle = np.where(distance_from_center<obstacle_radius)
        #    pressureCells_Mask[j_obstacle,i_obstacle] = True
            
            sd = "2nd-order-central"
            DivGradp = spatial_operators.create_DivGrad_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask)
            DivGradu = spatial_operators.create_DivGrad_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
            DivGradv = spatial_operators.create_DivGrad_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"periodic")
            delxu = spatial_operators.create_delx_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,sd,"periodic")
            delxp = spatial_operators.create_delx_operator(Dxu,Dyu,Xu,Yu,pressureCells_Mask,sd,"periodic")
            delyu = spatial_operators.create_dely_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,sd,"periodic")
            delxv = spatial_operators.create_delx_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,sd,"periodic")
            delyv = spatial_operators.create_dely_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,sd,"periodic")
            divxu = spatial_operators.create_divx_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,sd)
            divyv = spatial_operators.create_divy_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,sd)
            divxp = spatial_operators.create_divx_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,sd)
            divyp = spatial_operators.create_divy_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,sd)
             #initial values of u,v,p
            u_initial = U(0,Xu,Yu)
            v_initial = V(0,Xv,Yv)
            p_initial = P(0,Xp,Yp)
            u_final = U(1,Xu,Yu)
            v_final = V(1,Xv,Yv)
            p_final = P(1,Xp,Yp)
            u = u_initial
            v = v_initial
            p = p_initial
            u_star = np.zeros(u.size)
            v_star = np.zeros(v.size)
            time_t = 0
            it = 1
            
            
            #Flattening u,v,p 2-D arrays
            u_flattened = u.flatten()
            v_flattened = v.flatten()
            u_star_flattened = u_star.flatten()
            v_star_flattened = v_star.flatten()
            p_flattened = p.flatten()
        
            while time_t <= t_final:
                #Finding viscous terms in prediction
                
                R_viscous_u = nu*DivGradu*u_flattened
                R_viscous_v = nu*DivGradv*v_flattened
                
                #interpolation of v
                
                v_temp = np.c_[v[:,-1],v,v[:,0]]
                v_inter_1 = 0.5*(v_temp[:-1,:]+v_temp[1:,:])
                v_inter_2 = 0.5*(v_inter_1[:,:-1]+v_inter_1[:,1:])
                v_inter_2_flattened = v_inter_2.flatten()
                
                #interpolation of u
                
                u_temp = np.r_[[u[-1,:]],u,[u[0,:]]]
                u_inter_1 = 0.5*(u_temp[:,:-1]+u_temp[:,1:])
                u_inter_2 = 0.5*(u_inter_1[:-1,:]+u_inter_1[1:,:])
                u_inter_2_flattened = u_inter_2.flatten()
                
                #Convective term
                R_convective_u = u_flattened*delxu*u_flattened + v_inter_2_flattened*delyu*u_flattened
                R_convective_v = u_inter_2_flattened*delxv*v_flattened + v_flattened*delyv*v_flattened
                
                #Prediction step
                
                u_star_flattened = u_flattened + dt*(-R_convective_u + R_viscous_u)
                v_star_flattened = v_flattened + dt*(-R_convective_v + R_viscous_v)
                
                #Pressure-Velocity Coupling
                
                Div = (1./dt)*(divxu*u_star_flattened+divyv*v_star_flattened)
                p_flattened = spysparselinalg.spsolve(DivGradp,Div)
                
                #correction step
                
                #Reshaping into 2-D arrays
                u_star = np.reshape(u_star_flattened,(Nxc,Nyc+1))
                v_star = np.reshape(v_star_flattened,(Nxc+1,Nyc))
                
                #Done to preserve the boundary values, which should not enter the correction step as that is only done for u,v in the interior.
                u = u_star
                v = v_star
                
                #Redefines u_star and v_star as all faces excluding the boundaries, note that boundaries are still stored in u and v
                #Corrections for core
                u_star = u_star[:,1:-1]
                v_star = v_star[1:-1,:]
                u_star_flattened = u_star.flatten()
                v_star_flattened = v_star.flatten()
                u_star_flattened = u_star_flattened - dt*divxp*p_flattened
                v_star_flattened = v_star_flattened - dt*divyp*p_flattened
                
                #Reshaping the core into a 2-D array and putting it inside the core of u and v. Note that the boundaries are already present.
                u_star = np.reshape(u_star_flattened,(Nxc,Nyc-1))
                v_star = np.reshape(v_star_flattened,(Nxc-1,Nyc))
                u[:,1:-1] = u_star
                v[1:-1,:] = v_star
                print "Time:%f" %(time_t)
                time_t += dt
                it += it
                
            e_TR_u_1[index] = np.linalg.norm(u-u_final)/np.sqrt(u.size)
            e_TR_v_1[index] = np.linalg.norm(v-v_final)/np.sqrt(v.size)
            print "For Nx = %d, Ny = %d:" %(Nxc,Nyc)
            print "RMS error for u: %2.10f" %(e_TR_u_1[index])
            print "RMS error for v: %2.10f" %(e_TR_v_1[index])
            
            
    #            # Plot solution. For code validation
    #        fig = plt.figure(0, figsize=(figwidth,figheight))
    #        ax   = fig.add_axes([0.15,0.15,0.8,0.8])
    #        plt.axes(ax)
    #        plt.contourf(Xu[1:-1,1:-1],Yu[1:-1,1:-1],u)
    #        plt.colorbar()
    #        
    #        fig2 = plt.figure(1, figsize=(figwidth,figheight))
    #        ax   = fig2.add_axes([0.15,0.15,0.8,0.8])
    #        plt.axes(ax)
    #        plt.contourf(Xu[1:-1,1:-1],Yu[1:-1,1:-1],u_final)
    #        plt.colorbar()
    #        
    #        fig3 = plt.figure(2, figsize=(figwidth,figheight))
    #        ax   = fig3.add_axes([0.15,0.15,0.8,0.8])
    #        plt.axes(ax)
    #        plt.contourf(Xv[1:-1,1:-1],Yv[1:-1,1:-1],v)
    #        plt.colorbar()
    #        
    #        fig4 = plt.figure(3, figsize=(figwidth,figheight))
    #        ax   = fig4.add_axes([0.15,0.15,0.8,0.8])
    #        plt.axes(ax)
    #        plt.contourf(Xv[1:-1,1:-1],Yv[1:-1,1:-1],v_final)
    #        plt.colorbar()
    #        
    #        ax.grid('on',which='both')
    #        plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    #        plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    #        ax.set_xlabel(r"$x$",fontsize=textFontSize)
    #        ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
    #        plt.axis("tight")
    #        plt.axis("equal")
    #        
    #        plt.show()
            
#        #Plotter code for RMS error
#        figure_name = "RMS error vs No. of grid points for problem 2 for RK1.pdf"
#        fig = plt.figure(0, figsize=(figwidth,figheight))
#        plt.title("RMS error for u for RK1")
#        plt.xlabel(r"N",fontsize=textFontSize)
#        plt.ylabel(r"$epsilon_{RMS}$",fontsize=textFontSize,rotation=90)
#        plt.plot(NXC,e_TR_u,'b--',label="RMS error for u")
#        plt.plot(NXC,e_TR_v,'r--',label="RMS error for v")
#        plt.legend(loc='best')
#        figure_file_path = figure_folder + figure_name
#        print "Saving figure: " + figure_file_path
#        plt.tight_layout()
#        plt.savefig(figure_file_path)
#        plt.close()
        
    if RK2:
        e_TR_u_2 = np.zeros(NXC.size)
        e_TR_v_2 = np.zeros(NYC.size)
        for index,Nxc in enumerate(NXC):
            Nyc = int(NYC[index])
            Nxc = int(Nxc)
            print "Nxc = %d, Nyc = %d" %(Nxc,Nyc)
                    # number of (pressure) cells = mass conservation cells
            #########################################
            ######## Preprocessing Stage ############
            
            # You might have to include ghost-cells here
            # Depending on your application
            
            # define grid for u and v velocity components first
            # and then define pressure cells locations
            xsi_u = np.linspace(0.,1.0,Nxc+1)
            xsi_v = np.linspace(0.,1.0,Nyc+1)
            # uniform grid
            xu = xsi_u*Lx
            yv = xsi_v*Ly
            # non-uniform grid can be specified by redefining xsi_u and xsi_v. For e.g.
            #xu = (xsi_u**4.)*Lx # can be replaced by non-uniform grid
            #yv = (xsi_v**4.)*Ly # can be replaced by non-uniform grid
            
            # creating ghost cells
            #internodal distance at the beginning and end
            dxu0 = np.diff(xu)[0]
            dxuL = np.diff(xu)[-1]
            #Concatenate the ghost points (obtained by flipping the same distances on either side of the first and last point)
            xu = np.concatenate([[xu[0]-dxu0],xu,[xu[-1]+dxuL]])
            
            dyv0 = np.diff(yv)[0]
            dyvL = np.diff(yv)[-1]
            yv = np.concatenate([[yv[0]-dyv0],yv,[yv[-1]+dyvL]])
            
            
            #Pressure nodes are cell centres
            #Velocity nodes described above are staggered wrt pressure nodes.
            dxc = np.diff(xu)  # pressure-cells spacings
            dyc = np.diff(yv)
        
            
            #Pressure points
            xc = 0.5*(xu[:-1]+xu[1:])  # total of Nxc cells
            yc = 0.5*(yv[:-1]+yv[1:])  # total of Nyc cells
            
            dxu = np.concatenate([[0],np.diff(xc),[0]])
            dyv = np.concatenate([[0],np.diff(yc),[0]])
            
            # note that indexing is Xc[j_y,i_x] or Xc[j,i]
            
            #Pressure grid
            [Xp,Yp]   = np.meshgrid(xc,yc)     # these arrays are with ghost cells
            [Dxp,Dyp] = np.meshgrid(dxc,dyc)   # these arrays are with ghost cells
            
            #U grid
            [Xu,Yu] = np.meshgrid(xu,yc)
            [Dxu,Dyu] = np.meshgrid(dxu,dyc)
            
            #V grid
            [Xv,Yv] = np.meshgrid(xc,yv)
            [Dxv,Dyv] = np.meshgrid(dxc,dyv)
            
            ### familiarize yourself with 'flattening' options
            # phi_Fordering     = Phi.flatten(order='F') # F (column-major), Fortran/Matlab default
            # phi_Cordering     = Phi.flatten(order='C') # C (row-major), C/Python default
            # phi_PythonDefault = Phi.flatten()          # Python default
            # compare Phi[:,0] and Phi[0,:] with phi[:Nxc]
            
            #Mask has been flipped as compared to the starter code.
            #Mask is designed to be false for all points where pressure is to be calculated i.e. Mask==True means pressure is masked at those regions
            #This required changing np.zeros to np.ones in line 71
            # Pre-allocated False = fluid points
            pressureCells_Mask = np.ones(Xp.shape)
            pressureCells_Mask[1:-1,1:-1] = False # note that ghost cells are also marked true as a default value of 1 is assigned
            
            u_velocityCells_Mask = np.ones(Xu.shape)
            u_velocityCells_Mask[1:-1,1:-1] = False
            v_velocityCells_Mask = np.ones(Xv.shape)
            v_velocityCells_Mask [1:-1,1:-1] = False
            # Introducing obstacle in pressure Mask
        #    obstacle_radius = 0.0*Lx # set equal to 0.0*Lx to remove obstacle
        #    distance_from_center = np.sqrt(np.power(Xp-Lx/2.,2.0)+np.power(Yp-Ly/2.,2.0))
        #    j_obstacle,i_obstacle = np.where(distance_from_center<obstacle_radius)
        #    pressureCells_Mask[j_obstacle,i_obstacle] = True
            
            sd = "2nd-order-central"
            DivGradp = spatial_operators.create_DivGrad_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask)
            DivGradu = spatial_operators.create_DivGrad_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
            DivGradv = spatial_operators.create_DivGrad_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"periodic")
            delxu = spatial_operators.create_delx_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,sd,"periodic")
            delxp = spatial_operators.create_delx_operator(Dxu,Dyu,Xu,Yu,pressureCells_Mask,sd,"periodic")
            delyu = spatial_operators.create_dely_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,sd,"periodic")
            delxv = spatial_operators.create_delx_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,sd,"periodic")
            delyv = spatial_operators.create_dely_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,sd,"periodic")
            divxu = spatial_operators.create_divx_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,sd)
            divyv = spatial_operators.create_divy_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,sd)
            divxp = spatial_operators.create_divx_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,sd)
            divyp = spatial_operators.create_divy_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,sd)
             #initial values of u,v,p
            u_initial = U(0,Xu,Yu)
            v_initial = V(0,Xv,Yv)
            p_initial = P(0,Xp,Yp)
            u_final = U(1,Xu,Yu)
            v_final = V(1,Xv,Yv)
            p_final = P(1,Xp,Yp)
            u = u_initial
            v = v_initial
            p = p_initial
            u_star = np.zeros(u.size)
            v_star = np.zeros(v.size)
            time_t = 0
            it = 1
            
            
            #Flattening u,v,p 2-D arrays
            u_flattened = u.flatten()
            v_flattened = v.flatten()
            u_star_flattened = u_star.flatten()
            v_star_flattened = v_star.flatten()
            p_flattened = p.flatten()
            
            while time_t <= t_final:
            
            #Step 1
                #Finding viscous terms in prediction
                
                R_viscous_u = nu*DivGradu*u_flattened
                R_viscous_v = nu*DivGradv*v_flattened
                
                #interpolation of v
                
                v_temp = np.c_[v[:,-1],v,v[:,0]]
                v_inter_1 = 0.5*(v_temp[:-1,:]+v_temp[1:,:])
                v_inter_2 = 0.5*(v_inter_1[:,:-1]+v_inter_1[:,1:])
                v_inter_2_flattened = v_inter_2.flatten()
                
                #interpolation of u
                
                u_temp = np.r_[[u[-1,:]],u,[u[0,:]]]
                u_inter_1 = 0.5*(u_temp[:,:-1]+u_temp[:,1:])
                u_inter_2 = 0.5*(u_inter_1[:-1,:]+u_inter_1[1:,:])
                u_inter_2_flattened = u_inter_2.flatten()
                
                #Convective term
                R_convective_u = u_flattened*delxu*u_flattened + v_inter_2_flattened*delyu*u_flattened
                R_convective_v = u_inter_2_flattened*delxv*v_flattened + v_flattened*delyv*v_flattened
                
                #1st-Prediction step
                
                u_star_flattened = u_flattened + dt*(-R_convective_u + R_viscous_u)
                v_star_flattened = v_flattened + dt*(-R_convective_v + R_viscous_v)
                
            #2nd-step
                
                #Re-cast u_star and v_star in 2-D array form for interpolation
                
                u_star = np.reshape(u_star_flattened,(Nxc,Nyc+1))
                v_star = np.reshape(v_star_flattened,(Nxc+1,Nyc))
                
                #interpolation of v
                
                v_temp = np.c_[v_star[:,-1],v_star,v_star[:,0]]
                v_inter_1 = 0.5*(v_temp[:-1,:]+v_temp[1:,:])
                v_inter_2 = 0.5*(v_inter_1[:,:-1]+v_inter_1[:,1:])
                v_inter_2_flattened = v_inter_2.flatten()
                
                #interpolation of u
                
                u_temp = np.r_[[u_star[-1,:]],u_star,[u_star[0,:]]]
                u_inter_1 = 0.5*(u_temp[:,:-1]+u_temp[:,1:])
                u_inter_2 = 0.5*(u_inter_1[:-1,:]+u_inter_1[1:,:])
                u_inter_2_flattened = u_inter_2.flatten()
                
                R_convective_u_1 = u_star_flattened*delxu*u_star_flattened + v_inter_2_flattened*delyu*u_star_flattened
                R_convective_v_1 = u_inter_2_flattened*delxv*v_flattened + v_flattened*delyv*v_star_flattened
                R_viscous_u_1 = nu*DivGradu*u_star_flattened
                R_viscous_v_1 = nu*DivGradv*v_star_flattened
                
                #2nd Prediction step
                
                u_star_flattened = u_flattened + (dt/2.0)*(-R_convective_u_1 + R_viscous_u_1)
                v_star_flattened = v_flattened + (dt/2.0)*(-R_convective_v_1 + R_viscous_v_1)
                
                #Pressure-Velocity Coupling
                
                Div = (1./dt)*(divxu*u_star_flattened+divyv*v_star_flattened)
                p_flattened = spysparselinalg.spsolve(DivGradp,Div)
                
                #correction step
                
                #Reshaping into 2-D arrays
                u_star = np.reshape(u_star_flattened,(Nxc,Nyc+1))
                v_star = np.reshape(v_star_flattened,(Nxc+1,Nyc))
                
                #Done to preserve the boundary values, which should not enter the correction step as that is only done for u,v in the interior.
                u = u_star
                v = v_star
                
                #Redefines u_star and v_star as all faces excluding the boundaries, note that boundaries are still stored in u and v
                #Corrections for core
                u_star = u_star[:,1:-1]
                v_star = v_star[1:-1,:]
                u_star_flattened = u_star.flatten()
                v_star_flattened = v_star.flatten()
                u_star_flattened = u_star_flattened - dt*divxp*p_flattened
                v_star_flattened = v_star_flattened - dt*divyp*p_flattened
                
                #Reshaping the core into a 2-D array and putting it inside the core of u and v. Note that the boundaries are already present.
                u_star = np.reshape(u_star_flattened,(Nxc,Nyc-1))
                v_star = np.reshape(v_star_flattened,(Nxc-1,Nyc))
                u[:,1:-1] = u_star
                v[1:-1,:] = v_star
                print "Time:%f" %(time_t)
                time_t += dt
                it += it
                
            e_TR_u_2[index] = np.linalg.norm(u-u_final)/np.sqrt(u.size)
            e_TR_v_2[index] = np.linalg.norm(v-v_final)/np.sqrt(v.size)
            print "For Nx = %d, Ny = %d:" %(Nxc,Nyc)
            print "RMS error for u: %2.10f" %(e_TR_u_2[index])
            print "RMS error for v: %2.10f" %(e_TR_v_2[index])
            
#        #Plotter code for RMS error
#        figure_name = "RMS error vs No. of grid points for problem 2 for RK2.pdf"
#        fig = plt.figure(1, figsize=(figwidth,figheight))
#        plt.title("RMS error for u for RK2")
#        plt.xlabel(r"N",fontsize=textFontSize)
#        plt.ylabel(r"$epsilon_{RMS}$",fontsize=textFontSize,rotation=90)
#        plt.plot(NXC,e_TR_u,'b--',label="RMS error for u")
#        plt.plot(NXC,e_TR_v,'r--',label="RMS error for v")
#        plt.legend(loc='best')
#        figure_file_path = figure_folder + figure_name
#        print "Saving figure: " + figure_file_path
#        plt.tight_layout()
#        plt.savefig(figure_file_path)
#        plt.close()
            
#            # Plot solution. For code validation
#        fig = plt.figure(0, figsize=(figwidth,figheight))
#        ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#        plt.axes(ax)
#        plt.contourf(Xu[1:-1,1:-1],Yu[1:-1,1:-1],u)
#        plt.colorbar()
#        
#        fig2 = plt.figure(1, figsize=(figwidth,figheight))
#        ax   = fig2.add_axes([0.15,0.15,0.8,0.8])
#        plt.axes(ax)
#        plt.contourf(Xu[1:-1,1:-1],Yu[1:-1,1:-1],u_final)
#        plt.colorbar()
#        
#        fig3 = plt.figure(2, figsize=(figwidth,figheight))
#        ax   = fig3.add_axes([0.15,0.15,0.8,0.8])
#        plt.axes(ax)
#        plt.contourf(Xv[1:-1,1:-1],Yv[1:-1,1:-1],v)
#        plt.colorbar()
#        
#        fig4 = plt.figure(3, figsize=(figwidth,figheight))
#        ax   = fig4.add_axes([0.15,0.15,0.8,0.8])
#        plt.axes(ax)
#        plt.contourf(Xv[1:-1,1:-1],Yv[1:-1,1:-1],v_final)
#        plt.colorbar()
#        
#        ax.grid('on',which='both')
#        plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
#        plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
#        ax.set_xlabel(r"$x$",fontsize=textFontSize)
#        ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
#        plt.axis("tight")
#        plt.axis("equal")
#        
#        plt.show()
        
    if RK4:
        e_TR_u_4 = np.zeros(NXC.size)
        e_TR_v_4 = np.zeros(NYC.size)
        for index,Nxc in enumerate(NXC):
            Nyc = int(NYC[index])
            Nxc = int(Nxc)
            print "Nxc = %d, Nyc = %d" %(Nxc,Nyc)
                    # number of (pressure) cells = mass conservation cells
            #########################################
            ######## Preprocessing Stage ############
            
            # You might have to include ghost-cells here
            # Depending on your application
            
            # define grid for u and v velocity components first
            # and then define pressure cells locations
            xsi_u = np.linspace(0.,1.0,Nxc+1)
            xsi_v = np.linspace(0.,1.0,Nyc+1)
            # uniform grid
            xu = xsi_u*Lx
            yv = xsi_v*Ly
            # non-uniform grid can be specified by redefining xsi_u and xsi_v. For e.g.
            #xu = (xsi_u**4.)*Lx # can be replaced by non-uniform grid
            #yv = (xsi_v**4.)*Ly # can be replaced by non-uniform grid
            
            # creating ghost cells
            #internodal distance at the beginning and end
            dxu0 = np.diff(xu)[0]
            dxuL = np.diff(xu)[-1]
            #Concatenate the ghost points (obtained by flipping the same distances on either side of the first and last point)
            xu = np.concatenate([[xu[0]-dxu0],xu,[xu[-1]+dxuL]])
            
            dyv0 = np.diff(yv)[0]
            dyvL = np.diff(yv)[-1]
            yv = np.concatenate([[yv[0]-dyv0],yv,[yv[-1]+dyvL]])
            
            
            #Pressure nodes are cell centres
            #Velocity nodes described above are staggered wrt pressure nodes.
            dxc = np.diff(xu)  # pressure-cells spacings
            dyc = np.diff(yv)
        
            
            #Pressure points
            xc = 0.5*(xu[:-1]+xu[1:])  # total of Nxc cells
            yc = 0.5*(yv[:-1]+yv[1:])  # total of Nyc cells
            
            dxu = np.concatenate([[0],np.diff(xc),[0]])
            dyv = np.concatenate([[0],np.diff(yc),[0]])
            
            # note that indexing is Xc[j_y,i_x] or Xc[j,i]
            
            #Pressure grid
            [Xp,Yp]   = np.meshgrid(xc,yc)     # these arrays are with ghost cells
            [Dxp,Dyp] = np.meshgrid(dxc,dyc)   # these arrays are with ghost cells
            
            #U grid
            [Xu,Yu] = np.meshgrid(xu,yc)
            [Dxu,Dyu] = np.meshgrid(dxu,dyc)
            
            #V grid
            [Xv,Yv] = np.meshgrid(xc,yv)
            [Dxv,Dyv] = np.meshgrid(dxc,dyv)
            
            ### familiarize yourself with 'flattening' options
            # phi_Fordering     = Phi.flatten(order='F') # F (column-major), Fortran/Matlab default
            # phi_Cordering     = Phi.flatten(order='C') # C (row-major), C/Python default
            # phi_PythonDefault = Phi.flatten()          # Python default
            # compare Phi[:,0] and Phi[0,:] with phi[:Nxc]
            
            #Mask has been flipped as compared to the starter code.
            #Mask is designed to be false for all points where pressure is to be calculated i.e. Mask==True means pressure is masked at those regions
            #This required changing np.zeros to np.ones in line 71
            # Pre-allocated False = fluid points
            pressureCells_Mask = np.ones(Xp.shape)
            pressureCells_Mask[1:-1,1:-1] = False # note that ghost cells are also marked true as a default value of 1 is assigned
            
            u_velocityCells_Mask = np.ones(Xu.shape)
            u_velocityCells_Mask[1:-1,1:-1] = False
            v_velocityCells_Mask = np.ones(Xv.shape)
            v_velocityCells_Mask [1:-1,1:-1] = False
            # Introducing obstacle in pressure Mask
        #    obstacle_radius = 0.0*Lx # set equal to 0.0*Lx to remove obstacle
        #    distance_from_center = np.sqrt(np.power(Xp-Lx/2.,2.0)+np.power(Yp-Ly/2.,2.0))
        #    j_obstacle,i_obstacle = np.where(distance_from_center<obstacle_radius)
        #    pressureCells_Mask[j_obstacle,i_obstacle] = True
            
            sd = "2nd-order-central"
            DivGradp = spatial_operators.create_DivGrad_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask)
            DivGradu = spatial_operators.create_DivGrad_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
            DivGradv = spatial_operators.create_DivGrad_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"periodic")
            delxu = spatial_operators.create_delx_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,sd,"periodic")
            delxp = spatial_operators.create_delx_operator(Dxu,Dyu,Xu,Yu,pressureCells_Mask,sd,"periodic")
            delyu = spatial_operators.create_dely_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,sd,"periodic")
            delxv = spatial_operators.create_delx_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,sd,"periodic")
            delyv = spatial_operators.create_dely_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,sd,"periodic")
            divxu = spatial_operators.create_divx_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,sd)
            divyv = spatial_operators.create_divy_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,sd)
            divxp = spatial_operators.create_divx_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,sd)
            divyp = spatial_operators.create_divy_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,sd)
             #initial values of u,v,p
            u_initial = U(0,Xu,Yu)
            v_initial = V(0,Xv,Yv)
            p_initial = P(0,Xp,Yp)
            u_final = U(1,Xu,Yu)
            v_final = V(1,Xv,Yv)
            p_final = P(1,Xp,Yp)
            u = u_initial
            v = v_initial
            p = p_initial
            u_star = np.zeros(u.size)
            v_star = np.zeros(v.size)
            time_t = 0
            it = 1
            
            
            #Flattening u,v,p 2-D arrays
            u_flattened = u.flatten()
            v_flattened = v.flatten()
            u_star_flattened = u_star.flatten()
            v_star_flattened = v_star.flatten()
            p_flattened = p.flatten()
            
            while time_t <= t_final:
                
            #Step 1
                
                
                #interpolation of v
                
                v_temp = np.c_[v[:,-1],v,v[:,0]]
                v_inter_1 = 0.5*(v_temp[:-1,:]+v_temp[1:,:])
                v_inter_2 = 0.5*(v_inter_1[:,:-1]+v_inter_1[:,1:])
                v_inter_2_flattened = v_inter_2.flatten()
                
                #interpolation of u
                
                u_temp = np.r_[[u[-1,:]],u,[u[0,:]]]
                u_inter_1 = 0.5*(u_temp[:,:-1]+u_temp[:,1:])
                u_inter_2 = 0.5*(u_inter_1[:-1,:]+u_inter_1[1:,:])
                u_inter_2_flattened = u_inter_2.flatten()
                
                #Finding viscous terms in prediction
                
                R_viscous_u = nu*DivGradu*u_flattened
                R_viscous_v = nu*DivGradv*v_flattened
                
                #Convective term
                R_convective_u = u_flattened*delxu*u_flattened + v_inter_2_flattened*delyu*u_flattened
                R_convective_v = u_inter_2_flattened*delxv*v_flattened + v_flattened*delyv*v_flattened
                
                #1st-Prediction step
                
                u_star_flattened = u_flattened + (dt/2.0)*(-R_convective_u + R_viscous_u)
                v_star_flattened = v_flattened + (dt/2.0)*(-R_convective_v + R_viscous_v)
                
            #2nd-step
                
                #Re-cast u_star and v_star in 2-D array form for interpolation
                
                u_star = np.reshape(u_star_flattened,(Nxc,Nyc+1))
                v_star = np.reshape(v_star_flattened,(Nxc+1,Nyc))
                
                #interpolation of v
                
                v_temp = np.c_[v_star[:,-1],v_star,v_star[:,0]]
                v_inter_1 = 0.5*(v_temp[:-1,:]+v_temp[1:,:])
                v_inter_2 = 0.5*(v_inter_1[:,:-1]+v_inter_1[:,1:])
                v_inter_2_flattened = v_inter_2.flatten()
                
                #interpolation of u
                
                u_temp = np.r_[[u_star[-1,:]],u_star,[u_star[0,:]]]
                u_inter_1 = 0.5*(u_temp[:,:-1]+u_temp[:,1:])
                u_inter_2 = 0.5*(u_inter_1[:-1,:]+u_inter_1[1:,:])
                u_inter_2_flattened = u_inter_2.flatten()
                
                #Re-evaluate Viscous and convective terms
                R_convective_u_1 = u_star_flattened*delxu*u_star_flattened + v_inter_2_flattened*delyu*u_star_flattened
                R_convective_v_1 = u_inter_2_flattened*delxv*v_flattened + v_flattened*delyv*v_star_flattened
                R_viscous_u_1 = nu*DivGradu*u_star_flattened
                R_viscous_v_1 = nu*DivGradv*v_star_flattened
                
                #2nd Prediction step
                
                u_star_flattened = u_flattened + (dt/2.0)*(-R_convective_u_1 + R_viscous_u_1)
                v_star_flattened = v_flattened + (dt/2.0)*(-R_convective_v_1 + R_viscous_v_1)
                
             #3rd-step
                
                #Re-cast u_star and v_star in 2-D array form for interpolation
                
                u_star = np.reshape(u_star_flattened,(Nxc,Nyc+1))
                v_star = np.reshape(v_star_flattened,(Nxc+1,Nyc))
                
                #interpolation of v
                
                v_temp = np.c_[v_star[:,-1],v_star,v_star[:,0]]
                v_inter_1 = 0.5*(v_temp[:-1,:]+v_temp[1:,:])
                v_inter_2 = 0.5*(v_inter_1[:,:-1]+v_inter_1[:,1:])
                v_inter_2_flattened = v_inter_2.flatten()
                
                #interpolation of u
                
                u_temp = np.r_[[u_star[-1,:]],u_star,[u_star[0,:]]]
                u_inter_1 = 0.5*(u_temp[:,:-1]+u_temp[:,1:])
                u_inter_2 = 0.5*(u_inter_1[:-1,:]+u_inter_1[1:,:])
                u_inter_2_flattened = u_inter_2.flatten()
                
                #Re-evaluate Viscous and convective terms
                R_convective_u_2 = u_star_flattened*delxu*u_star_flattened + v_inter_2_flattened*delyu*u_star_flattened
                R_convective_v_2 = u_inter_2_flattened*delxv*v_flattened + v_flattened*delyv*v_star_flattened
                R_viscous_u_2 = nu*DivGradu*u_star_flattened
                R_viscous_v_2 = nu*DivGradv*v_star_flattened
                
                #3rd Prediction step
                
                u_star_flattened = u_flattened + dt*(-R_convective_u_2 + R_viscous_u_2)
                v_star_flattened = v_flattened + dt*(-R_convective_v_2 + R_viscous_v_2)
                
            #4th-step
                
                #Re-cast u_star and v_star in 2-D array form for interpolation
                
                u_star = np.reshape(u_star_flattened,(Nxc,Nyc+1))
                v_star = np.reshape(v_star_flattened,(Nxc+1,Nyc))
                
                #interpolation of v
                
                v_temp = np.c_[v_star[:,-1],v_star,v_star[:,0]]
                v_inter_1 = 0.5*(v_temp[:-1,:]+v_temp[1:,:])
                v_inter_2 = 0.5*(v_inter_1[:,:-1]+v_inter_1[:,1:])
                v_inter_2_flattened = v_inter_2.flatten()
                
                #interpolation of u
                
                u_temp = np.r_[[u_star[-1,:]],u_star,[u_star[0,:]]]
                u_inter_1 = 0.5*(u_temp[:,:-1]+u_temp[:,1:])
                u_inter_2 = 0.5*(u_inter_1[:-1,:]+u_inter_1[1:,:])
                u_inter_2_flattened = u_inter_2.flatten()
                
                #Re-evaluate Viscous and convective terms
                R_convective_u_3 = u_star_flattened*delxu*u_star_flattened + v_inter_2_flattened*delyu*u_star_flattened
                R_convective_v_3 = u_inter_2_flattened*delxv*v_flattened + v_flattened*delyv*v_star_flattened
                R_viscous_u_3 = nu*DivGradu*u_star_flattened
                R_viscous_v_3 = nu*DivGradv*v_star_flattened
                
                #Final Prediction
                
                u_star_flattened = u_flattened + (dt/6.0)*((-R_convective_u + R_viscous_u) + (2*(-R_convective_u_1 - R_convective_u_2 + R_viscous_u_1 + R_viscous_u_2)) + (-R_convective_u_3 + R_viscous_u_3))
                v_star_flattened = v_flattened + (dt/6.0)*((-R_convective_v + R_viscous_v) + (2*(-R_convective_v_1 - R_convective_v_2 + R_viscous_v_1 + R_viscous_v_2)) + (-R_convective_v_3 + R_viscous_v_3))
                
                #Pressure-Velocity Coupling
                
                Div = (1./dt)*(divxu*u_star_flattened+divyv*v_star_flattened)
                p_flattened = spysparselinalg.spsolve(DivGradp,Div)
                
                #correction step
                
                #Reshaping into 2-D arrays
                u_star = np.reshape(u_star_flattened,(Nxc,Nyc+1))
                v_star = np.reshape(v_star_flattened,(Nxc+1,Nyc))
                
                #Done to preserve the boundary values, which should not enter the correction step as that is only done for u,v in the interior.
                u = u_star
                v = v_star
                
                #Redefines u_star and v_star as all faces excluding the boundaries, note that boundaries are still stored in u and v
                #Corrections for core
                u_star = u_star[:,1:-1]
                v_star = v_star[1:-1,:]
                u_star_flattened = u_star.flatten()
                v_star_flattened = v_star.flatten()
                u_star_flattened = u_star_flattened - dt*divxp*p_flattened
                v_star_flattened = v_star_flattened - dt*divyp*p_flattened
                
                #Reshaping the core into a 2-D array and putting it inside the core of u and v. Note that the boundaries are already present.
                u_star = np.reshape(u_star_flattened,(Nxc,Nyc-1))
                v_star = np.reshape(v_star_flattened,(Nxc-1,Nyc))
                u[:,1:-1] = u_star
                v[1:-1,:] = v_star
                print "Time:%f" %(time_t)
                time_t += dt
                it += it
                
            e_TR_u_4[index] = np.linalg.norm(u-u_final)/np.sqrt(u.size)
            e_TR_v_4[index] = np.linalg.norm(v-v_final)/np.sqrt(v.size)
            print "For Nx = %d, Ny = %d:" %(Nxc,Nyc)
            print "RMS error for u: %2.10f" %(e_TR_u_4[index])
            print "RMS error for v: %2.10f" %(e_TR_v_4[index])
            
#        #Plotter code for RMS error
#        figure_name = "RMS error vs No. of grid points for problem 2 for RK4.pdf"
#        fig = plt.figure(2, figsize=(figwidth,figheight))
#        plt.title("RMS error for u for RK4")
#        plt.xlabel(r"N",fontsize=textFontSize)
#        plt.ylabel(r"$epsilon_{RMS}$",fontsize=textFontSize,rotation=90)
#        plt.plot(NXC,e_TR_u,'b--',label="RMS error for u")
#        plt.plot(NXC,e_TR_v,'r--',label="RMS error for v")
#        plt.legend(loc='best')
#        figure_file_path = figure_folder + figure_name
#        print "Saving figure: " + figure_file_path
#        plt.tight_layout()
#        plt.savefig(figure_file_path)
#        plt.close()
        
#        sru = np.linalg.norm(u-u_final)/np.sqrt(u.size)
#        srv = np.linalg.norm(v-v_final)/np.sqrt(v.size)
#            # Plot solution. For code validation
#        fig = plt.figure(0, figsize=(figwidth,figheight))
#        ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#        plt.axes(ax)
#        plt.contourf(Xu[1:-1,1:-1],Yu[1:-1,1:-1],u)
#        plt.colorbar()
#        
#        fig2 = plt.figure(1, figsize=(figwidth,figheight))
#        ax   = fig2.add_axes([0.15,0.15,0.8,0.8])
#        plt.axes(ax)
#        plt.contourf(Xu[1:-1,1:-1],Yu[1:-1,1:-1],u_final)
#        plt.colorbar()
#        
#        fig3 = plt.figure(2, figsize=(figwidth,figheight))
#        ax   = fig3.add_axes([0.15,0.15,0.8,0.8])
#        plt.axes(ax)
#        plt.contourf(Xv[1:-1,1:-1],Yv[1:-1,1:-1],v)
#        plt.colorbar()
#        
#        fig4 = plt.figure(3, figsize=(figwidth,figheight))
#        ax   = fig4.add_axes([0.15,0.15,0.8,0.8])
#        plt.axes(ax)
#        plt.contourf(Xv[1:-1,1:-1],Yv[1:-1,1:-1],v_final)
#        plt.colorbar()
#        
#        ax.grid('on',which='both')
#        plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
#        plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
#        ax.set_xlabel(r"$x$",fontsize=textFontSize)
#        ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
#        plt.axis("tight")
#        plt.axis("equal")
#        
#        plt.show()
        
    if RK3:
        e_TR_u_3 = np.zeros(NXC.size)
        e_TR_v_3 = np.zeros(NYC.size)
        for index,Nxc in enumerate(NXC):
            Nyc = int(NYC[index])
            Nxc = int(Nxc)
            print "Nxc = %d, Nyc = %d" %(Nxc,Nyc)
                    # number of (pressure) cells = mass conservation cells
            #########################################
            ######## Preprocessing Stage ############
            
            # You might have to include ghost-cells here
            # Depending on your application
            
            # define grid for u and v velocity components first
            # and then define pressure cells locations
            xsi_u = np.linspace(0.,1.0,Nxc+1)
            xsi_v = np.linspace(0.,1.0,Nyc+1)
            # uniform grid
            xu = xsi_u*Lx
            yv = xsi_v*Ly
            # non-uniform grid can be specified by redefining xsi_u and xsi_v. For e.g.
            #xu = (xsi_u**4.)*Lx # can be replaced by non-uniform grid
            #yv = (xsi_v**4.)*Ly # can be replaced by non-uniform grid
            
            # creating ghost cells
            #internodal distance at the beginning and end
            dxu0 = np.diff(xu)[0]
            dxuL = np.diff(xu)[-1]
            #Concatenate the ghost points (obtained by flipping the same distances on either side of the first and last point)
            xu = np.concatenate([[xu[0]-dxu0],xu,[xu[-1]+dxuL]])
            
            dyv0 = np.diff(yv)[0]
            dyvL = np.diff(yv)[-1]
            yv = np.concatenate([[yv[0]-dyv0],yv,[yv[-1]+dyvL]])
            
            
            #Pressure nodes are cell centres
            #Velocity nodes described above are staggered wrt pressure nodes.
            dxc = np.diff(xu)  # pressure-cells spacings
            dyc = np.diff(yv)
        
            
            #Pressure points
            xc = 0.5*(xu[:-1]+xu[1:])  # total of Nxc cells
            yc = 0.5*(yv[:-1]+yv[1:])  # total of Nyc cells
            
            dxu = np.concatenate([[0],np.diff(xc),[0]])
            dyv = np.concatenate([[0],np.diff(yc),[0]])
            
            # note that indexing is Xc[j_y,i_x] or Xc[j,i]
            
            #Pressure grid
            [Xp,Yp]   = np.meshgrid(xc,yc)     # these arrays are with ghost cells
            [Dxp,Dyp] = np.meshgrid(dxc,dyc)   # these arrays are with ghost cells
            
            #U grid
            [Xu,Yu] = np.meshgrid(xu,yc)
            [Dxu,Dyu] = np.meshgrid(dxu,dyc)
            
            #V grid
            [Xv,Yv] = np.meshgrid(xc,yv)
            [Dxv,Dyv] = np.meshgrid(dxc,dyv)
            
            ### familiarize yourself with 'flattening' options
            # phi_Fordering     = Phi.flatten(order='F') # F (column-major), Fortran/Matlab default
            # phi_Cordering     = Phi.flatten(order='C') # C (row-major), C/Python default
            # phi_PythonDefault = Phi.flatten()          # Python default
            # compare Phi[:,0] and Phi[0,:] with phi[:Nxc]
            
            #Mask has been flipped as compared to the starter code.
            #Mask is designed to be false for all points where pressure is to be calculated i.e. Mask==True means pressure is masked at those regions
            #This required changing np.zeros to np.ones in line 71
            # Pre-allocated False = fluid points
            pressureCells_Mask = np.ones(Xp.shape)
            pressureCells_Mask[1:-1,1:-1] = False # note that ghost cells are also marked true as a default value of 1 is assigned
            
            u_velocityCells_Mask = np.ones(Xu.shape)
            u_velocityCells_Mask[1:-1,1:-1] = False
            v_velocityCells_Mask = np.ones(Xv.shape)
            v_velocityCells_Mask [1:-1,1:-1] = False
            # Introducing obstacle in pressure Mask
        #    obstacle_radius = 0.0*Lx # set equal to 0.0*Lx to remove obstacle
        #    distance_from_center = np.sqrt(np.power(Xp-Lx/2.,2.0)+np.power(Yp-Ly/2.,2.0))
        #    j_obstacle,i_obstacle = np.where(distance_from_center<obstacle_radius)
        #    pressureCells_Mask[j_obstacle,i_obstacle] = True
            
            sd = "2nd-order-central"
            DivGradp = spatial_operators.create_DivGrad_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask)
            DivGradu = spatial_operators.create_DivGrad_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
            DivGradv = spatial_operators.create_DivGrad_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"periodic")
            delxu = spatial_operators.create_delx_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,sd,"periodic")
            delxp = spatial_operators.create_delx_operator(Dxu,Dyu,Xu,Yu,pressureCells_Mask,sd,"periodic")
            delyu = spatial_operators.create_dely_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,sd,"periodic")
            delxv = spatial_operators.create_delx_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,sd,"periodic")
            delyv = spatial_operators.create_dely_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,sd,"periodic")
            divxu = spatial_operators.create_divx_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,sd)
            divyv = spatial_operators.create_divy_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,sd)
            divxp = spatial_operators.create_divx_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,sd)
            divyp = spatial_operators.create_divy_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,sd)
             #initial values of u,v,p
            u_initial = U(0,Xu,Yu)
            v_initial = V(0,Xv,Yv)
            p_initial = P(0,Xp,Yp)
            u_final = U(1,Xu,Yu)
            v_final = V(1,Xv,Yv)
            p_final = P(1,Xp,Yp)
            u = u_initial
            v = v_initial
            p = p_initial
            u_star = np.zeros(u.size)
            v_star = np.zeros(v.size)
            time_t = 0
            it = 1
            a = 0.25
            A = 2./3.
            b = 3./20.
            B = 5./12.
            c = 3./5.
            
            
            #Flattening u,v,p 2-D arrays
            u_flattened = u.flatten()
            v_flattened = v.flatten()
            u_star_flattened = u_star.flatten()
            v_star_flattened = v_star.flatten()
            p_flattened = p.flatten()
            while time_t <= t_final:
                
            #Step 1
                
                
                #interpolation of v
                
                v_temp = np.c_[v[:,-1],v,v[:,0]]
                v_inter_1 = 0.5*(v_temp[:-1,:]+v_temp[1:,:])
                v_inter_2 = 0.5*(v_inter_1[:,:-1]+v_inter_1[:,1:])
                v_inter_2_flattened = v_inter_2.flatten()
                
                #interpolation of u
                
                u_temp = np.r_[[u[-1,:]],u,[u[0,:]]]
                u_inter_1 = 0.5*(u_temp[:,:-1]+u_temp[:,1:])
                u_inter_2 = 0.5*(u_inter_1[:-1,:]+u_inter_1[1:,:])
                u_inter_2_flattened = u_inter_2.flatten()
                
                #Finding viscous terms in prediction
                
                R_viscous_u = nu*DivGradu*u_flattened
                R_viscous_v = nu*DivGradv*v_flattened
                
                #Convective term
                R_convective_u = u_flattened*delxu*u_flattened + v_inter_2_flattened*delyu*u_flattened
                R_convective_v = u_inter_2_flattened*delxv*v_flattened + v_flattened*delyv*v_flattened
                
                #1st-Prediction step
                
                u_star_flattened_1 = u_flattened + (dt*a)*(-R_convective_u + R_viscous_u)
                v_star_flattened_1 = v_flattened + (dt*a)*(-R_convective_v + R_viscous_v)
                
            #2nd-step
                
                u_star_flattened_2 = u_flattened + (dt*A)*(-R_convective_u + R_viscous_u)
                v_star_flattened_2 = v_flattened + (dt*A)*(-R_convective_v + R_viscous_v)
                
             #3rd-step
                
                #Re-cast u_star and v_star in 2-D array form for interpolation
                
                u_star_2 = np.reshape(u_star_flattened_2,(Nxc,Nyc+1))
                v_star_2 = np.reshape(v_star_flattened_2,(Nxc+1,Nyc))
                
                #interpolation of v
                
                v_temp = np.c_[v_star_2[:,-1],v_star_2,v_star_2[:,0]]
                v_inter_1 = 0.5*(v_temp[:-1,:]+v_temp[1:,:])
                v_inter_2 = 0.5*(v_inter_1[:,:-1]+v_inter_1[:,1:])
                v_inter_2_flattened = v_inter_2.flatten()
                
                #interpolation of u
                
                u_temp = np.r_[[u_star_2[-1,:]],u_star_2,[u_star_2[0,:]]]
                u_inter_1 = 0.5*(u_temp[:,:-1]+u_temp[:,1:])
                u_inter_2 = 0.5*(u_inter_1[:-1,:]+u_inter_1[1:,:])
                u_inter_2_flattened = u_inter_2.flatten()
                
                #Re-evaluate Viscous and convective terms
                R_convective_u_2 = u_star_flattened_2*delxu*u_star_flattened_2 + v_inter_2_flattened*delyu*u_star_flattened_2
                R_convective_v_2 = u_inter_2_flattened*delxv*v_star_flattened_2 + v_star_flattened_2*delyv*v_star_flattened_2
                R_viscous_u_2 = nu*DivGradu*u_star_flattened_2
                R_viscous_v_2 = nu*DivGradv*v_star_flattened_2
                
                #3rd Prediction step
                
                u_star_flattened_3 = u_star_flattened_1 + (dt*b)*(-R_convective_u_2 + R_viscous_u_2)
                v_star_flattened_3 = v_star_flattened_1 + (dt*b)*(-R_convective_v_2 + R_viscous_v_2)
                
                #4th step
                
                u_star_flattened_4 = u_star_flattened_1 + (dt*B)*(-R_convective_u_2 + R_viscous_u_2)
                v_star_flattened_4 = v_star_flattened_1 + (dt*B)*(-R_convective_v_2 + R_viscous_v_2)
                
                #Re-cast u_star and v_star in 2-D array form for interpolation
                
                u_star_4 = np.reshape(u_star_flattened_4,(Nxc,Nyc+1))
                v_star_4 = np.reshape(v_star_flattened_4,(Nxc+1,Nyc))
                
                #interpolation of v
                
                v_temp = np.c_[v_star_4[:,-1],v_star_4,v_star_4[:,0]]
                v_inter_1 = 0.5*(v_temp[:-1,:]+v_temp[1:,:])
                v_inter_2 = 0.5*(v_inter_1[:,:-1]+v_inter_1[:,1:])
                v_inter_2_flattened = v_inter_2.flatten()
                
                #interpolation of u
                
                u_temp = np.r_[[u_star_4[-1,:]],u_star_4,[u_star_4[0,:]]]
                u_inter_1 = 0.5*(u_temp[:,:-1]+u_temp[:,1:])
                u_inter_2 = 0.5*(u_inter_1[:-1,:]+u_inter_1[1:,:])
                u_inter_2_flattened = u_inter_2.flatten()
                
                #Re-evaluate Viscous and convective terms
                R_convective_u_4 = u_star_flattened_2*delxu*u_star_flattened_4 + v_inter_2_flattened*delyu*u_star_flattened_4
                R_convective_v_4 = u_inter_2_flattened*delxv*v_star_flattened_4 + v_star_flattened_2*delyv*v_star_flattened_4
                R_viscous_u_4 = nu*DivGradu*u_star_flattened_4
                R_viscous_v_4 = nu*DivGradv*v_star_flattened_4
                
                #Final prediction
                
                u_star_flattened = u_star_flattened_3 + (dt*c)*(-R_convective_u_4 + R_viscous_u_4)
                v_star_flattened = v_star_flattened_3 + (dt*c)*(-R_convective_v_4 + R_viscous_v_4)
                
                #Pressure-Velocity Coupling
                
                Div = (1./dt)*(divxu*u_star_flattened+divyv*v_star_flattened)
                p_flattened = spysparselinalg.spsolve(DivGradp,Div)
                
                #correction step
                
                #Reshaping into 2-D arrays
                u_star = np.reshape(u_star_flattened,(Nxc,Nyc+1))
                v_star = np.reshape(v_star_flattened,(Nxc+1,Nyc))
                
                #Done to preserve the boundary values, which should not enter the correction step as that is only done for u,v in the interior.
                u = u_star
                v = v_star
                
                #Redefines u_star and v_star as all faces excluding the boundaries, note that boundaries are still stored in u and v
                #Corrections for core
                u_star = u_star[:,1:-1]
                v_star = v_star[1:-1,:]
                u_star_flattened = u_star.flatten()
                v_star_flattened = v_star.flatten()
                u_star_flattened = u_star_flattened - dt*divxp*p_flattened
                v_star_flattened = v_star_flattened - dt*divyp*p_flattened
                
                #Reshaping the core into a 2-D array and putting it inside the core of u and v. Note that the boundaries are already present.
                u_star = np.reshape(u_star_flattened,(Nxc,Nyc-1))
                v_star = np.reshape(v_star_flattened,(Nxc-1,Nyc))
                u[:,1:-1] = u_star
                v[1:-1,:] = v_star
    #            print "Iterartion: %d" %(it)
                print "Time:%f" %(time_t)
                time_t += dt
                it += it
                
            e_TR_u_3[index] = np.linalg.norm(u-u_final)/np.sqrt(u.size)
            e_TR_v_3[index] = np.linalg.norm(v-v_final)/np.sqrt(v.size)
            print "For Nx = %d, Ny = %d:" %(Nxc,Nyc)
            print "RMS error for u: %2.10f" %(e_TR_u_3[index])
            print "RMS error for v: %2.10f" %(e_TR_v_3[index])
            
#        #Plotter code for RMS error
#        figure_name = "RMS error vs No. of grid points for problem 2 for RK3.pdf"
#        fig = plt.figure(3, figsize=(figwidth,figheight))
#        plt.title("RMS error for u for RK3")
#        plt.xlabel(r"N",fontsize=textFontSize)
#        plt.ylabel(r"$epsilon_{RMS}$",fontsize=textFontSize,rotation=90)
#        plt.plot(NXC,e_TR_u_3,'b--',label="RMS error for u")
#        plt.plot(NXC,e_TR_v_3,'r--',label="RMS error for v")
#        plt.legend(loc='best')
#        figure_file_path = figure_folder + figure_name
#        print "Saving figure: " + figure_file_path
#        plt.tight_layout()
#        plt.savefig(figure_file_path)
#        plt.close()
            
            
#                # Plot solution. For code validation
#            fig = plt.figure(0, figsize=(figwidth,figheight))
#            ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#            plt.axes(ax)
#            plt.contourf(Xu[1:-1,1:-1],Yu[1:-1,1:-1],u)
#            plt.colorbar()
#            
#            fig2 = plt.figure(1, figsize=(figwidth,figheight))
#            ax   = fig2.add_axes([0.15,0.15,0.8,0.8])
#            plt.axes(ax)
#            plt.contourf(Xu[1:-1,1:-1],Yu[1:-1,1:-1],u_final)
#            plt.colorbar()
#            
#            fig3 = plt.figure(2, figsize=(figwidth,figheight))
#            ax   = fig3.add_axes([0.15,0.15,0.8,0.8])
#            plt.axes(ax)
#            plt.contourf(Xv[1:-1,1:-1],Yv[1:-1,1:-1],v)
#            plt.colorbar()
#            
#            fig4 = plt.figure(3, figsize=(figwidth,figheight))
#            ax   = fig4.add_axes([0.15,0.15,0.8,0.8])
#            plt.axes(ax)
#            plt.contourf(Xv[1:-1,1:-1],Yv[1:-1,1:-1],v_final)
#            plt.colorbar()
#            
#            ax.grid('on',which='both')
#            plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
#            plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
#            ax.set_xlabel(r"$x$",fontsize=textFontSize)
#            ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
#            plt.axis("tight")
#            plt.axis("equal")
#            
#            plt.show()
            
            
            
    #Plotter code
    figure_name = "RMS error for u vs Grid spacing for problem 2.pdf"
    fig = plt.figure(0, figsize=(figwidth,figheight))
    plt.title("RMS error for u")
    plt.xlabel(r"Grid spacing",fontsize=textFontSize)
    plt.ylabel(r"$epsilon_{RMS}$",fontsize=textFontSize,rotation=90)
    plt.loglog(2*np.pi/NXC,e_TR_u_1,'b--',label="RK1")
    plt.loglog(2*np.pi/NXC,e_TR_u_2,'r--',label="RK2")
    plt.loglog(2*np.pi/NXC,e_TR_u_3,'k--',label="RK3")
    plt.loglog(2*np.pi/NXC,e_TR_u_4,'g--',label="RK4")
    plt.loglog(2*np.pi/NXC,(2*np.pi/NXC)**2,'k-',label="Order 2")
    plt.loglog(2*np.pi/NXC,(2*np.pi/NXC)**3,'b-',label="Order 3")
    plt.loglog(2*np.pi/NXC,(2*np.pi/NXC)**1,'r-',label="Order 1")
    plt.legend(loc='best')
    figure_file_path = figure_folder + figure_name
    print "Saving figure: " + figure_file_path
    plt.tight_layout()
    plt.savefig(figure_file_path)
    plt.close()
    
    figure_name = "RMS error of v vs Grid spacing for problem 2.pdf"
    fig = plt.figure(0, figsize=(figwidth,figheight))
    plt.title("RMS error for v")
    plt.xlabel(r"Grid spacing",fontsize=textFontSize)
    plt.ylabel(r"$epsilon_{RMS}$",fontsize=textFontSize,rotation=90)
    plt.loglog(2*np.pi/NXC,e_TR_v_1,'b--',label="RK1")
    plt.loglog(2*np.pi/NXC,e_TR_v_2,'r--',label="RK2")
    plt.loglog(2*np.pi/NXC,e_TR_v_3,'k--',label="RK3")
    plt.loglog(2*np.pi/NXC,e_TR_v_4,'g--',label="RK4")
    plt.loglog(2*np.pi/NXC,(2*np.pi/NXC)**2,'k-',label="Order 2")
    plt.loglog(2*np.pi/NXC,(2*np.pi/NXC)**3,'b-',label="Order 3")
    plt.loglog(2*np.pi/NXC,(2*np.pi/NXC)**1,'r-',label="Order 1")
    plt.legend(loc='best')
    figure_file_path = figure_folder + figure_name
    print "Saving figure: " + figure_file_path
    plt.tight_layout()
    plt.savefig(figure_file_path)
    plt.close()
    
        
        

        
        
        
        
    
        
    
