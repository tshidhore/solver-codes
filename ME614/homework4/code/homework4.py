import os
import sys
import numpy as np
import scipy.sparse as scysparse
from pdb import set_trace as keyboard
from time import sleep
import spatial_operators
import spatial_operators_TG
import scipy.sparse as scysparse
import scipy.sparse.linalg as spysparselinalg  # sparse linear algebra
import scipy.linalg as scylinalg               # non-sparse linear algebra
import pylab as plt
from matplotlib import rc as matplotlibrc
import time # has the equivalent of tic/toc



###Put brackets in matrix multiplication to let thw solver know in which order is the matrix multiplication desired (see lid driven cavity RK2)
machine_epsilon = np.finfo(float).eps
matplotlibrc('text.latex', preamble='\usepackage{color}')
matplotlibrc('text',usetex=True)
matplotlibrc('font', family='serif')

def U(t,X,Y):
    u = np.exp(-2*t)*np.cos(X[1:-1,1:-1])*np.sin(Y[1:-1,1:-1])
    return u
    
def V(t,X,Y):
    v = -np.exp(-2*t)*np.sin(X[1:-1,1:-1])*np.cos(Y[1:-1,1:-1])
    return v
    
def P(t,X,Y):
    p = -0.25*np.exp(-4*t)*(np.cos(2*X[1:-1,1:-1])+np.cos(2*Y[1:-1,1:-1]))
    return p
    
    
def ContourPlot_TG(X,Y,var,fig_num=0):
    fig = plt.figure(fig_num, figsize=(figwidth,figheight))
    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
    plt.axes(ax)
    plt.contourf(X[1:-1,1:-1],Y[1:-1,1:-1],var,100)
    plt.colorbar()
    plt.show()
    
    
def Grid_Generate_TG(Nxc,Nyc,Lx,Ly):
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
#            xu = (Lx/2)*((np.tanh(alp*(2*xsi_u - 1.))/np.tanh(alp)) + 1.)
#            yv = (Ly/2)*((np.tanh(alp*(2*xsi_v - 1.))/np.tanh(alp)) + 1.)
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
        u_velocityCells_Mask[1:-1,1:-2] = False
        v_velocityCells_Mask = np.ones(Xv.shape)
        v_velocityCells_Mask [1:-2,1:-1] = False
        
        # Introducing obstacle in pressure Mask
    #    obstacle_radius = 0.0*Lx # set equal to 0.0*Lx to remove obstacle
    #    distance_from_center = np.sqrt(np.power(Xp-Lx/2.,2.0)+np.power(Yp-Ly/2.,2.0))
    #    j_obstacle,i_obstacle = np.where(distance_from_center<obstacle_radius)
    #    pressureCells_Mask[j_obstacle,i_obstacle] = True
        
        return Xp,Dxp,Yp,Dyp,Xu,Dxu,Yu,Dyu,Xv,Dxv,Yv,Dyv,pressureCells_Mask,u_velocityCells_Mask,v_velocityCells_Mask
    
def SD2RK1_TG(Nxc,Nyc,Lx,Ly,t_final,dt,Xp,Dxp,Yp,Dyp,Xu,Dxu,Yu,Dyu,Xv,Dxv,Yv,Dyv,pressureCells_Mask,u_velocityCells_Mask,v_velocityCells_Mask):    
        
    DivGradp = spatial_operators_TG.create_DivGrad_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,"periodic")
    DivGradp[0,:] = 0.
    DivGradp[0,0] = 1.0
    DivGradu = spatial_operators_TG.create_DivGrad_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
    DivGradu = spatial_operators_TG.create_DivGrad_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
    DivGradv = spatial_operators_TG.create_DivGrad_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"periodic")
    delxu = spatial_operators_TG.create_delx_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
    delyu = spatial_operators_TG.create_dely_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
    delxv = spatial_operators_TG.create_delx_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"periodic")
    delyv = spatial_operators_TG.create_dely_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"periodic")
    divxu = spatial_operators_TG.create_divx_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
    divyv = spatial_operators_TG.create_divy_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"periodic")
    divxp = spatial_operators_TG.create_divx_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,"periodic")
    divyp = spatial_operators_TG.create_divy_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,"periodic")
    
#            keyboard()
     #initial values of u,v,p
    u_initial = U(0,Xu,Yu)
    v_initial = V(0,Xv,Yv)
    p_initial = P(0,Xp,Yp)
    u_final = U(t_final,Xu,Yu)
    v_final = V(t_final,Xv,Yv)
#    p_final = P(t_final,Xp,Yp)
    
    u = u_initial
    v = v_initial
    p = p_initial
    
    
    time_t = 0
    it = 1
    
 
    #Flattening u,v,p 2-D arrays
#            print p
    while time_t <= t_final:
        
        #Flattening u,v,p 2-D arrays
        u_flattened = u[:,0:-1].flatten()
        v_flattened = v[0:-1,:].flatten()
        u_star = np.zeros(u[:,0:-1].shape)
        v_star = np.zeros(v[0:-1,:].shape)
        u_star_flattened = u_star.flatten()
        v_star_flattened = v_star.flatten()
        p = np.zeros(p_initial.size)
        p_flattened = p.flatten()
        #Finding viscous terms in prediction
        
        R_viscous_u = DivGradu*u_flattened
        R_viscous_v = DivGradv*v_flattened
        
        #interpolation of v
        
        v_temp = np.c_[v[:,-1],v]
        v_inter_1 = 0.5*(v_temp[:-1,:]+v_temp[1:,:])
        v_inter_2 = 0.5*(v_inter_1[:,:-1]+v_inter_1[:,1:])
        v_inter_2_flattened = v_inter_2.flatten()
        
        #interpolation of u
        
        u_temp = np.r_[[u[-1,:]],u]
        u_inter_1 = 0.5*(u_temp[:,:-1]+u_temp[:,1:])
        u_inter_2 = 0.5*(u_inter_1[:-1,:]+u_inter_1[1:,:])
        u_inter_2_flattened = u_inter_2.flatten()
        
        #Convective term
        R_convective_u = u_flattened*(delxu*u_flattened) + v_inter_2_flattened*(delyu*u_flattened)
        R_convective_v = u_inter_2_flattened*(delxv*v_flattened) + v_flattened*(delyv*v_flattened)
        
        #Prediction step
        u_star_flattened = u_flattened + dt*(-R_convective_u + R_viscous_u)
        v_star_flattened = v_flattened + dt*(-R_convective_v + R_viscous_v)
        
#                keyboard()
        u_star = np.reshape(u_star_flattened,(Nxc,Nyc))
        v_star = np.reshape(v_star_flattened,(Nxc,Nyc))
        
        u_star = np.c_[u_star,u_star[:,0]]
        v_star = np.r_[v_star,[v_star[0,:]]]
#                keyboard()
#                u_star[:,-1] = u_star[:,0]
#                v_star[-1,:] = v_star[0,:]
        
        #Pressure-Velocity Coupling
        
#                keyboard()
        Div_u_star = ((divxu*u_star_flattened) + (divyv*v_star_flattened))/dt
        Div_u_star[0] = P(time_t,Xp,Yp)[0,0]
        p_flattened = spysparselinalg.spsolve(DivGradp,Div_u_star)
#                print np.max(p_flattened)
        p = np.reshape(p_flattened,(Nxc,Nyc))
#                p = p[:,:] - p[Nxc/2,Nyc/2]  #Anchoring middle value to 0
#                print p_flattened[5]
#                print p_final_flattened[5]
        
        #correction step
        
#                #Reshaping into 2-D arrays
#                u_star = np.reshape(u_star_flattened,(Nxc,Nyc+1))
#                v_star = np.reshape(v_star_flattened,(Nxc+1,Nyc))
        
#                #Done to preserve the boundary values, which should not enter the correction step as that is only done for u,v in the interior.
#                u = u_star
#                v = v_star
        
        #Redefines u_star and v_star as all faces excluding the boundaries, note that boundaries are still stored in u and v
        #Corrections for core
        u_star_corr = u_star[:,1:]
        v_star_corr = v_star[1:,:]
#                keyboard()
        u_star_corr_flattened = u_star_corr.flatten()
        v_star_corr_flattened = v_star_corr.flatten()
        corr_u = dt*(divxp*p_flattened)
        corr_v = dt*(divyp*p_flattened)
#                keyboard()
        u_star_corr_flattened = u_star_corr_flattened - corr_u
        v_star_corr_flattened = v_star_corr_flattened - corr_v
        
#                #Reshaping the core into a 2-D array and putting it inside the core of u and v. Note that the boundaries are already present.
#                u_star = np.reshape(u_star_flattened,(Nxc,Nyc-1))
#                v_star = np.reshape(v_star_flattened,(Nxc-1,Nyc))
        u = u_initial
        v = v_initial
        u[:,1:] = np.reshape(u_star_corr_flattened,(Nxc,Nyc))
        v[1:,:] = np.reshape(v_star_corr_flattened,(Nxc,Nyc))
#                keyboard()
        u[:,0] = u[:,-1]
#                u[:,0] = u_star[:,0]
        v[0,:] = v[-1,:]
#                v_star[-1,:] = v_star[-1,:]
        time_t += dt
        print "Time:%f" %(time_t)
        it += it
        
    e_TR_u = np.linalg.norm(u-u_final)/np.sqrt(u.size)
    e_TR_v = np.linalg.norm(v-v_final)/np.sqrt(v.size)
    print "For Nx = %d, Ny = %d:" %(Nxc,Nyc)
    print "RMS error for u: %2.10f" %(e_TR_u)
    print "RMS error for v: %2.10f" %(e_TR_v)
        
    return u,v,p,e_TR_u,e_TR_v
    
def SD2RK2_TG(Nxc,Nyc,Lx,Ly,t_final,dt,Xp,Dxp,Yp,Dyp,Xu,Dxu,Yu,Dyu,Xv,Dxv,Yv,Dyv,pressureCells_Mask,u_velocityCells_Mask,v_velocityCells_Mask):
    
    DivGradp = spatial_operators_TG.create_DivGrad_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,"periodic")
    DivGradp[0,:] = 0.
    DivGradp[0,0] = 1.0
    DivGradu = spatial_operators_TG.create_DivGrad_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
    DivGradu = spatial_operators_TG.create_DivGrad_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
    DivGradv = spatial_operators_TG.create_DivGrad_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"periodic")
    delxu = spatial_operators_TG.create_delx_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
    delyu = spatial_operators_TG.create_dely_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
    delxv = spatial_operators_TG.create_delx_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"periodic")
    delyv = spatial_operators_TG.create_dely_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"periodic")
    divxu = spatial_operators_TG.create_divx_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
    divyv = spatial_operators_TG.create_divy_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"periodic")
    divxp = spatial_operators_TG.create_divx_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,"periodic")
    divyp = spatial_operators_TG.create_divy_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,"periodic")
     #initial values of u,v,p
    u_initial = U(0,Xu,Yu)
    v_initial = V(0,Xv,Yv)
    p_initial = P(0,Xp,Yp)
    u_final = U(1,Xu,Yu)
    v_final = V(1,Xv,Yv)
#    p_final = P(1,Xp,Yp)
    
    u = u_initial
    v = v_initial
    p = p_initial
    
    time_t = 0
    it = 1
    
    
    while time_t <= t_final:
    
    #Step 1
    
        #Flattening u,v,p 2-D arrays
        u_flattened = u[:,0:-1].flatten()
        v_flattened = v[0:-1,:].flatten()
        u_star = np.zeros(u[:,0:-1].shape)
        v_star = np.zeros(v[0:-1,:].shape)
        u_star_flattened = u_star.flatten()
        v_star_flattened = v_star.flatten()
        p = np.zeros(p_initial.size)
        p_flattened = p.flatten()
        #Finding viscous terms in prediction
        
        R_viscous_u = DivGradu*u_flattened
        R_viscous_v = DivGradv*v_flattened
        
        #interpolation of v
        
        v_temp = np.c_[v[:,-1],v]
        v_inter_1 = 0.5*(v_temp[:-1,:]+v_temp[1:,:])
        v_inter_2 = 0.5*(v_inter_1[:,:-1]+v_inter_1[:,1:])
        v_inter_2_flattened = v_inter_2.flatten()
        
        #interpolation of u
        
        u_temp = np.r_[[u[-1,:]],u]
        u_inter_1 = 0.5*(u_temp[:,:-1]+u_temp[:,1:])
        u_inter_2 = 0.5*(u_inter_1[:-1,:]+u_inter_1[1:,:])
        u_inter_2_flattened = u_inter_2.flatten()
        
        #Convective term
        R_convective_u = u_flattened*(delxu*u_flattened) + v_inter_2_flattened*(delyu*u_flattened)
        R_convective_v = u_inter_2_flattened*(delxv*v_flattened) + v_flattened*(delyv*v_flattened)
        
        #1st-Prediction step
        
        u_star_flattened = u_flattened + dt*(-R_convective_u + R_viscous_u)
        v_star_flattened = v_flattened + dt*(-R_convective_v + R_viscous_v)
        
        u_star = np.reshape(u_star_flattened,(Nxc,Nyc))
        v_star = np.reshape(v_star_flattened,(Nxc,Nyc))
        
        u_star = np.c_[u_star,u_star[:,0]]
        v_star = np.r_[v_star,[v_star[0,:]]]
        
    #2nd-step
        
        #interpolation of v
        
        v_temp = np.c_[v_star[:,-1],v_star]
        v_inter_1 = 0.5*(v_temp[:-1,:]+v_temp[1:,:])
        v_inter_2 = 0.5*(v_inter_1[:,:-1]+v_inter_1[:,1:])
        v_inter_2_flattened = v_inter_2.flatten()
        
        #interpolation of u
        
        u_temp = np.r_[[u_star[-1,:]],u_star]
        u_inter_1 = 0.5*(u_temp[:,:-1]+u_temp[:,1:])
        u_inter_2 = 0.5*(u_inter_1[:-1,:]+u_inter_1[1:,:])
        u_inter_2_flattened = u_inter_2.flatten()
        
        u_star_flattened = u_star[:,0:-1].flatten()
        v_star_flattened = v_star[0:-1,:].flatten()
        
        R_convective_u_1 = u_star_flattened*(delxu*u_star_flattened) + v_inter_2_flattened*(delyu*u_star_flattened)
        R_convective_v_1 = u_inter_2_flattened*(delxv*v_star_flattened) + v_star_flattened*(delyv*v_star_flattened)
        R_viscous_u_1 = DivGradu*u_star_flattened
        R_viscous_v_1 = DivGradv*v_star_flattened
        
        
        
        #2nd Prediction step
        
        u_star_flattened = u_flattened + (dt/2.0)*(-R_convective_u_1 + R_viscous_u_1 - R_convective_u + R_viscous_u)
        v_star_flattened = v_flattened + (dt/2.0)*(-R_convective_v_1 + R_viscous_v_1 - R_convective_v + R_viscous_v)
        
        u_star = np.reshape(u_star_flattened,(Nxc,Nyc))
        v_star = np.reshape(v_star_flattened,(Nxc,Nyc))
        
        u_star = np.c_[u_star,u_star[:,0]]
        v_star = np.r_[v_star,[v_star[0,:]]]
        
        #Pressure-Velocity Coupling
        
#                keyboard()
        Div_u_star = ((divxu*u_star_flattened) + (divyv*v_star_flattened))/dt
        Div_u_star[0] = P(time_t,Xp,Yp)[0,0]
        p_flattened = spysparselinalg.spsolve(DivGradp,Div_u_star)
#                print np.max(p_flattened)
        p = np.reshape(p_flattened,(Nxc,Nyc))
#                p = p[:,:] - p[Nxc/2,Nyc/2]  #Anchoring middle value to 0
#                print p_flattened[5]
#                print p_final_flattened[5]
        
        #correction step
        
#                #Reshaping into 2-D arrays
#                u_star = np.reshape(u_star_flattened,(Nxc,Nyc+1))
#                v_star = np.reshape(v_star_flattened,(Nxc+1,Nyc))
        
#                #Done to preserve the boundary values, which should not enter the correction step as that is only done for u,v in the interior.
#                u = u_star
#                v = v_star
        
        #Redefines u_star and v_star as all faces excluding the boundaries, note that boundaries are still stored in u and v
        #Corrections for core
        u_star_corr = u_star[:,1:]
        v_star_corr = v_star[1:,:]
#                keyboard()
        u_star_corr_flattened = u_star_corr.flatten()
        v_star_corr_flattened = v_star_corr.flatten()
        corr_u = dt*(divxp*p_flattened)
        corr_v = dt*(divyp*p_flattened)
#                keyboard()
        u_star_corr_flattened = u_star_corr_flattened - corr_u
        v_star_corr_flattened = v_star_corr_flattened - corr_v
        
#                #Reshaping the core into a 2-D array and putting it inside the core of u and v. Note that the boundaries are already present.
#                u_star = np.reshape(u_star_flattened,(Nxc,Nyc-1))
#                v_star = np.reshape(v_star_flattened,(Nxc-1,Nyc))
        u = u_initial
        v = v_initial
        u[:,1:] = np.reshape(u_star_corr_flattened,(Nxc,Nyc))
        v[1:,:] = np.reshape(v_star_corr_flattened,(Nxc,Nyc))
#                keyboard()
        u[:,0] = u[:,-1]
#                u[:,0] = u_star[:,0]
        v[0,:] = v[-1,:]
#                v_star[-1,:] = v_star[-1,:]
        print "Time:%f" %(time_t)
        time_t += dt
        it += it
        
    e_TR_u = np.linalg.norm(u-u_final)/np.sqrt(u.size)
    e_TR_v = np.linalg.norm(v-v_final)/np.sqrt(v.size)
    print "For Nx = %d, Ny = %d:" %(Nxc,Nyc)
    print "RMS error for u: %2.10f" %(e_TR_u)
    print "RMS error for v: %2.10f" %(e_TR_v)
    
    return u,v,p,e_TR_u,e_TR_v
    
def SD2RK3_TG(Nxc,Nyc,Lx,Ly,t_final,dt,Xp,Dxp,Yp,Dyp,Xu,Dxu,Yu,Dyu,Xv,Dxv,Yv,Dyv,pressureCells_Mask,u_velocityCells_Mask,v_velocityCells_Mask):
    
    DivGradp = spatial_operators_TG.create_DivGrad_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,"periodic")
    DivGradp[0,:] = 0.
    DivGradp[0,0] = 1.0
    DivGradu = spatial_operators_TG.create_DivGrad_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
    DivGradu = spatial_operators_TG.create_DivGrad_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
    DivGradv = spatial_operators_TG.create_DivGrad_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"periodic")
    delxu = spatial_operators_TG.create_delx_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
    delyu = spatial_operators_TG.create_dely_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
    delxv = spatial_operators_TG.create_delx_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"periodic")
    delyv = spatial_operators_TG.create_dely_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"periodic")
    divxu = spatial_operators_TG.create_divx_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
    divyv = spatial_operators_TG.create_divy_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"periodic")
    divxp = spatial_operators_TG.create_divx_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,"periodic")
    divyp = spatial_operators_TG.create_divy_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,"periodic")
    
     #initial values of u,v,p
    u_initial = U(0,Xu,Yu)
    v_initial = V(0,Xv,Yv)
    p_initial = P(0,Xp,Yp)
    
    u_final = U(t_final,Xu,Yu)
    v_final = V(t_final,Xv,Yv)
    
    u = u_initial
    v = v_initial
    p = p_initial
    
    time_t = 0
    it = 1
    
    a = 0.25
    A = 2./3.
    b = 3./20.
    B = 5./12.
    c = 3./5.
    
    while time_t <= t_final:
                
    #Step 1
        
        #Flattening u,v,p 2-D arrays
        u_flattened = u[:,0:-1].flatten()
        v_flattened = v[0:-1,:].flatten()
        u_star = np.zeros(u[:,0:-1].shape)
        v_star = np.zeros(v[0:-1,:].shape)
        u_star_flattened = u_star.flatten()
        v_star_flattened = v_star.flatten()
        p = np.zeros(p_initial.size)
        p_flattened = p.flatten()

        #interpolation of v
        
        v_temp = np.c_[v[:,-1],v]
        v_inter_1 = 0.5*(v_temp[:-1,:]+v_temp[1:,:])
        v_inter_2 = 0.5*(v_inter_1[:,:-1]+v_inter_1[:,1:])
        v_inter_2_flattened = v_inter_2.flatten()
        
        #interpolation of u
        
        u_temp = np.r_[[u[-1,:]],u]
        u_inter_1 = 0.5*(u_temp[:,:-1]+u_temp[:,1:])
        u_inter_2 = 0.5*(u_inter_1[:-1,:]+u_inter_1[1:,:])
        u_inter_2_flattened = u_inter_2.flatten()
        
        #Finding viscous terms in prediction
        
        R_viscous_u = DivGradu*u_flattened
        R_viscous_v = DivGradv*v_flattened
        
        #Convective term
        R_convective_u = u_flattened*(delxu*u_flattened) + v_inter_2_flattened*(delyu*u_flattened)
        R_convective_v = u_inter_2_flattened*(delxv*v_flattened) + v_flattened*(delyv*v_flattened)
        
        #1st-Prediction step
        
        u_star_flattened_1 = u_flattened + (dt*a)*(-R_convective_u + R_viscous_u)
        v_star_flattened_1 = v_flattened + (dt*a)*(-R_convective_v + R_viscous_v)
        
    #2nd-step
        
        u_star_flattened_2 = u_flattened + (dt*A)*(-R_convective_u + R_viscous_u)
        v_star_flattened_2 = v_flattened + (dt*A)*(-R_convective_v + R_viscous_v)
        
     #3rd-step
        
        u_star_2 = np.reshape(u_star_flattened_2,(Nxc,Nyc))
        v_star_2 = np.reshape(v_star_flattened_2,(Nxc,Nyc))
        
        u_star_2 = np.c_[u_star_2,u_star_2[:,0]]
        v_star_2 = np.r_[v_star_2,[v_star_2[0,:]]]
        
        #interpolation of v
        
        v_temp = np.c_[v_star_2[:,-1],v_star_2]
        v_inter_1 = 0.5*(v_temp[:-1,:]+v_temp[1:,:])
        v_inter_2 = 0.5*(v_inter_1[:,:-1]+v_inter_1[:,1:])
        v_inter_2_flattened = v_inter_2.flatten()
        
        #interpolation of u
        
        u_temp = np.r_[[u_star_2[-1,:]],u_star_2]
        u_inter_1 = 0.5*(u_temp[:,:-1]+u_temp[:,1:])
        u_inter_2 = 0.5*(u_inter_1[:-1,:]+u_inter_1[1:,:])
        u_inter_2_flattened = u_inter_2.flatten()
        
        #Re-evaluate Viscous and convective terms
        R_convective_u_2 = u_star_flattened_2*(delxu*u_star_flattened_2) + v_inter_2_flattened*(delyu*u_star_flattened_2)
        R_convective_v_2 = u_inter_2_flattened*(delxv*v_star_flattened_2) + v_star_flattened_2*(delyv*v_star_flattened_2)
        R_viscous_u_2 = DivGradu*u_star_flattened_2
        R_viscous_v_2 = DivGradv*v_star_flattened_2
        
        #3rd Prediction step
        
        u_star_flattened_3 = u_star_flattened_1 + (dt*b)*(-R_convective_u_2 + R_viscous_u_2)
        v_star_flattened_3 = v_star_flattened_1 + (dt*b)*(-R_convective_v_2 + R_viscous_v_2)
        
        #4th step
        
        u_star_flattened_4 = u_star_flattened_1 + (dt*B)*(-R_convective_u_2 + R_viscous_u_2)
        v_star_flattened_4 = v_star_flattened_1 + (dt*B)*(-R_convective_v_2 + R_viscous_v_2)
        
        #Re-cast u_star and v_star in 2-D array form for interpolation
        
        u_star_4 = np.reshape(u_star_flattened_4,(Nxc,Nyc))
        v_star_4 = np.reshape(v_star_flattened_4,(Nxc,Nyc))
        
        u_star_4 = np.c_[u_star_4,u_star_4[:,0]]
        v_star_4 = np.r_[v_star_4,[v_star_4[0,:]]]
        
        #interpolation of v
        
        v_temp = np.c_[v_star_4[:,-1],v_star_4]
        v_inter_1 = 0.5*(v_temp[:-1,:]+v_temp[1:,:])
        v_inter_2 = 0.5*(v_inter_1[:,:-1]+v_inter_1[:,1:])
        v_inter_4_flattened = v_inter_2.flatten()
        
        #interpolation of u
        
        u_temp = np.r_[[u_star_4[-1,:]],u_star_4]
        u_inter_1 = 0.5*(u_temp[:,:-1]+u_temp[:,1:])
        u_inter_2 = 0.5*(u_inter_1[:-1,:]+u_inter_1[1:,:])
        u_inter_4_flattened = u_inter_2.flatten()
        
        #Re-evaluate Viscous and convective terms
        R_convective_u_4 = u_star_flattened_4*(delxu*u_star_flattened_4) + v_inter_4_flattened*(delyu*u_star_flattened_4)
        R_convective_v_4 = u_inter_4_flattened*(delxv*v_star_flattened_4) + v_star_flattened_4*(delyv*v_star_flattened_4)
        R_viscous_u_4 = DivGradu*u_star_flattened_4
        R_viscous_v_4 = DivGradv*v_star_flattened_4
        
        #Final prediction
        
        u_star_flattened = u_star_flattened_3 + (dt*c)*(-R_convective_u_4 + R_viscous_u_4)
        v_star_flattened = v_star_flattened_3 + (dt*c)*(-R_convective_v_4 + R_viscous_v_4)
        
        u_star = np.reshape(u_star_flattened,(Nxc,Nyc))
        v_star = np.reshape(v_star_flattened,(Nxc,Nyc))
        
        u_star = np.c_[u_star,u_star[:,0]]
        v_star = np.r_[v_star,[v_star[0,:]]]
        
        #Pressure-Velocity Coupling
        
#                keyboard()
        Div_u_star = ((divxu*u_star_flattened) + (divyv*v_star_flattened))/dt
        Div_u_star[0] = P(time_t,Xp,Yp)[0,0]
        p_flattened = spysparselinalg.spsolve(DivGradp,Div_u_star)
#                print np.max(p_flattened)
        p = np.reshape(p_flattened,(Nxc,Nyc))
#                p = p[:,:] - p[Nxc/2,Nyc/2]  #Anchoring middle value to 0
#                print p_flattened[5]
#                print p_final_flattened[5]
        
        #correction step
        
#                #Reshaping into 2-D arrays
#                u_star = np.reshape(u_star_flattened,(Nxc,Nyc+1))
#                v_star = np.reshape(v_star_flattened,(Nxc+1,Nyc))
        
#                #Done to preserve the boundary values, which should not enter the correction step as that is only done for u,v in the interior.
#                u = u_star
#                v = v_star
        
        #Redefines u_star and v_star as all faces excluding the boundaries, note that boundaries are still stored in u and v
        #Corrections for core
        u_star_corr = u_star[:,1:]
        v_star_corr = v_star[1:,:]
#                keyboard()
        u_star_corr_flattened = u_star_corr.flatten()
        v_star_corr_flattened = v_star_corr.flatten()
        corr_u = dt*(divxp*p_flattened)
        corr_v = dt*(divyp*p_flattened)
#                keyboard()
        u_star_corr_flattened = u_star_corr_flattened - corr_u
        v_star_corr_flattened = v_star_corr_flattened - corr_v
        
#                #Reshaping the core into a 2-D array and putting it inside the core of u and v. Note that the boundaries are already present.
#                u_star = np.reshape(u_star_flattened,(Nxc,Nyc-1))
#                v_star = np.reshape(v_star_flattened,(Nxc-1,Nyc))
        u = u_initial
        v = v_initial
        u[:,1:] = np.reshape(u_star_corr_flattened,(Nxc,Nyc))
        v[1:,:] = np.reshape(v_star_corr_flattened,(Nxc,Nyc))
#                keyboard()
        u[:,0] = u[:,-1]
#                u[:,0] = u_star[:,0]
        v[0,:] = v[-1,:]
#                v_star[-1,:] = v_star[-1,:]
        print "Time:%f" %(time_t)
        time_t += dt
        it += it
        
    e_TR_u = np.linalg.norm(u-u_final)/np.sqrt(u.size)
    e_TR_v = np.linalg.norm(v-v_final)/np.sqrt(v.size)
    print "For Nx = %d, Ny = %d:" %(Nxc,Nyc)
    print "RMS error for u: %2.10f" %(e_TR_u)
    print "RMS error for v: %2.10f" %(e_TR_v)
    
    return u,v,p,e_TR_u,e_TR_v
    
def SD2RK4_TG(Nxc,Nyc,Lx,Ly,t_final,dt,Xp,Dxp,Yp,Dyp,Xu,Dxu,Yu,Dyu,Xv,Dxv,Yv,Dyv,pressureCells_Mask,u_velocityCells_Mask,v_velocityCells_Mask):
    
    DivGradp = spatial_operators_TG.create_DivGrad_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,"periodic")
    DivGradp[0,:] = 0.
    DivGradp[0,0] = 1.0
    DivGradu = spatial_operators_TG.create_DivGrad_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
    DivGradu = spatial_operators_TG.create_DivGrad_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
    DivGradv = spatial_operators_TG.create_DivGrad_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"periodic")
    delxu = spatial_operators_TG.create_delx_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
    delyu = spatial_operators_TG.create_dely_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
    delxv = spatial_operators_TG.create_delx_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"periodic")
    delyv = spatial_operators_TG.create_dely_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"periodic")
    divxu = spatial_operators_TG.create_divx_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"periodic")
    divyv = spatial_operators_TG.create_divy_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"periodic")
    divxp = spatial_operators_TG.create_divx_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,"periodic")
    divyp = spatial_operators_TG.create_divy_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,"periodic")
    
     #initial values of u,v,p
    u_initial = U(0,Xu,Yu)
    v_initial = V(0,Xv,Yv)
    p_initial = P(0,Xp,Yp)
    
    u_final = U(t_final,Xu,Yu)
    v_final = V(t_final,Xv,Yv)
    
    u = u_initial
    v = v_initial
    p = p_initial
    
    time_t = 0
    it = 1
    
    
    while time_t <= t_final:
        
    #Step 1
        #Flattening u,v,p 2-D arrays
        u_flattened = u[:,0:-1].flatten()
        v_flattened = v[0:-1,:].flatten()
        u_star = np.zeros(u[:,0:-1].shape)
        v_star = np.zeros(v[0:-1,:].shape)
        u_star_flattened = u_star.flatten()
        v_star_flattened = v_star.flatten()
        p = np.zeros(p_initial.size)
        p_flattened = p.flatten()
        
        #interpolation of v
        
        v_temp = np.c_[v[:,-1],v]
        v_inter_1 = 0.5*(v_temp[:-1,:]+v_temp[1:,:])
        v_inter_2 = 0.5*(v_inter_1[:,:-1]+v_inter_1[:,1:])
        v_inter_2_flattened = v_inter_2.flatten()
        
        #interpolation of u
        
        u_temp = np.r_[[u[-1,:]],u]
        u_inter_1 = 0.5*(u_temp[:,:-1]+u_temp[:,1:])
        u_inter_2 = 0.5*(u_inter_1[:-1,:]+u_inter_1[1:,:])
        u_inter_2_flattened = u_inter_2.flatten()
        
        #Finding viscous terms in prediction
        
        R_viscous_u = DivGradu*u_flattened
        R_viscous_v = DivGradv*v_flattened
        
        #Convective term
        R_convective_u = u_flattened*(delxu*u_flattened) + v_inter_2_flattened*(delyu*u_flattened)
        R_convective_v = u_inter_2_flattened*(delxv*v_flattened) + v_flattened*(delyv*v_flattened)
        
        #1st-Prediction step
        
        u_star_flattened = u_flattened + (dt/2.0)*(-R_convective_u + R_viscous_u)
        v_star_flattened = v_flattened + (dt/2.0)*(-R_convective_v + R_viscous_v)
        
        u_star = np.reshape(u_star_flattened,(Nxc,Nyc))
        v_star = np.reshape(v_star_flattened,(Nxc,Nyc))
        
        u_star = np.c_[u_star,u_star[:,0]]
        v_star = np.r_[v_star,[v_star[0,:]]]
        
    #2nd-step
        
        #interpolation of v
        
        v_temp = np.c_[v_star[:,-1],v_star]
        v_inter_1 = 0.5*(v_temp[:-1,:]+v_temp[1:,:])
        v_inter_2 = 0.5*(v_inter_1[:,:-1]+v_inter_1[:,1:])
        v_inter_2_flattened = v_inter_2.flatten()
        
        #interpolation of u
        
        u_temp = np.r_[[u_star[-1,:]],u_star]
        u_inter_1 = 0.5*(u_temp[:,:-1]+u_temp[:,1:])
        u_inter_2 = 0.5*(u_inter_1[:-1,:]+u_inter_1[1:,:])
        u_inter_2_flattened = u_inter_2.flatten()
        
        u_star_flattened = u_star[:,0:-1].flatten()
        v_star_flattened = v_star[0:-1,:].flatten()
        
        #Re-evaluate Viscous and convective terms
        R_convective_u_1 = u_star_flattened*(delxu*u_star_flattened) + v_inter_2_flattened*(delyu*u_star_flattened)
        R_convective_v_1 = u_inter_2_flattened*(delxv*v_star_flattened) + v_star_flattened*(delyv*v_star_flattened)
        R_viscous_u_1 = DivGradu*u_star_flattened
        R_viscous_v_1 = DivGradv*v_star_flattened
        
        #2nd Prediction step
        
        u_star_flattened = u_flattened + (dt/2.0)*(-R_convective_u_1 + R_viscous_u_1)
        v_star_flattened = v_flattened + (dt/2.0)*(-R_convective_v_1 + R_viscous_v_1)
        
        u_star = np.reshape(u_star_flattened,(Nxc,Nyc))
        v_star = np.reshape(v_star_flattened,(Nxc,Nyc))
        
        u_star = np.c_[u_star,u_star[:,0]]
        v_star = np.r_[v_star,[v_star[0,:]]]
        
     #3rd-step
        
        #interpolation of v
        
        v_temp = np.c_[v_star[:,-1],v_star]
        v_inter_1 = 0.5*(v_temp[:-1,:]+v_temp[1:,:])
        v_inter_2 = 0.5*(v_inter_1[:,:-1]+v_inter_1[:,1:])
        v_inter_2_flattened = v_inter_2.flatten()
        
        #interpolation of u
        
        u_temp = np.r_[[u_star[-1,:]],u_star]
        u_inter_1 = 0.5*(u_temp[:,:-1]+u_temp[:,1:])
        u_inter_2 = 0.5*(u_inter_1[:-1,:]+u_inter_1[1:,:])
        u_inter_2_flattened = u_inter_2.flatten()
        
        u_star_flattened = u_star[:,0:-1].flatten()
        v_star_flattened = v_star[0:-1,:].flatten()
        
        
        #Re-evaluate Viscous and convective terms
        R_convective_u_2 = u_star_flattened*(delxu*u_star_flattened) + v_inter_2_flattened*(delyu*u_star_flattened)
        R_convective_v_2 = u_inter_2_flattened*(delxv*v_star_flattened) + v_star_flattened*(delyv*v_star_flattened)
        R_viscous_u_2 = DivGradu*u_star_flattened
        R_viscous_v_2 = DivGradv*v_star_flattened
        
        #3rd Prediction step
        
        u_star_flattened = u_flattened + dt*(-R_convective_u_2 + R_viscous_u_2)
        v_star_flattened = v_flattened + dt*(-R_convective_v_2 + R_viscous_v_2)
        
        u_star = np.reshape(u_star_flattened,(Nxc,Nyc))
        v_star = np.reshape(v_star_flattened,(Nxc,Nyc))
        
        u_star = np.c_[u_star,u_star[:,0]]
        v_star = np.r_[v_star,[v_star[0,:]]]
        
    #4th-step
        
        #interpolation of v
        
        v_temp = np.c_[v_star[:,-1],v_star]
        v_inter_1 = 0.5*(v_temp[:-1,:]+v_temp[1:,:])
        v_inter_2 = 0.5*(v_inter_1[:,:-1]+v_inter_1[:,1:])
        v_inter_2_flattened = v_inter_2.flatten()
        
        #interpolation of u
        
        u_temp = np.r_[[u_star[-1,:]],u_star]
        u_inter_1 = 0.5*(u_temp[:,:-1]+u_temp[:,1:])
        u_inter_2 = 0.5*(u_inter_1[:-1,:]+u_inter_1[1:,:])
        u_inter_2_flattened = u_inter_2.flatten()
        
        u_star_flattened = u_star[:,0:-1].flatten()
        v_star_flattened = v_star[0:-1,:].flatten()
        
        
        #Re-evaluate Viscous and convective terms
        R_convective_u_3 = u_star_flattened*(delxu*u_star_flattened) + v_inter_2_flattened*(delyu*u_star_flattened)
        R_convective_v_3 = u_inter_2_flattened*(delxv*v_star_flattened) + v_flattened*(delyv*v_star_flattened)
        R_viscous_u_3 = DivGradu*u_star_flattened
        R_viscous_v_3 = DivGradv*v_star_flattened
        
        #Final Prediction
        
        u_star_flattened = u_flattened + (dt/6.0)*((-R_convective_u + R_viscous_u) + (2*(-R_convective_u_1 - R_convective_u_2 + R_viscous_u_1 + R_viscous_u_2)) + (-R_convective_u_3 + R_viscous_u_3))
        v_star_flattened = v_flattened + (dt/6.0)*((-R_convective_v + R_viscous_v) + (2*(-R_convective_v_1 - R_convective_v_2 + R_viscous_v_1 + R_viscous_v_2)) + (-R_convective_v_3 + R_viscous_v_3))
        
        u_star = np.reshape(u_star_flattened,(Nxc,Nyc))
        v_star = np.reshape(v_star_flattened,(Nxc,Nyc))
        
        u_star = np.c_[u_star,u_star[:,0]]
        v_star = np.r_[v_star,[v_star[0,:]]]
        
        #Pressure-Velocity Coupling
        
#                keyboard()
        Div_u_star = ((divxu*u_star_flattened) + (divyv*v_star_flattened))/dt
        Div_u_star[0] = P(time_t,Xp,Yp)[0,0]
        p_flattened = spysparselinalg.spsolve(DivGradp,Div_u_star)
#                print np.max(p_flattened)
        p = np.reshape(p_flattened,(Nxc,Nyc))
#                p = p[:,:] - p[Nxc/2,Nyc/2]  #Anchoring middle value to 0
#                print p_flattened[5]
#                print p_final_flattened[5]
        
        #correction step
        
#                #Reshaping into 2-D arrays
#                u_star = np.reshape(u_star_flattened,(Nxc,Nyc+1))
#                v_star = np.reshape(v_star_flattened,(Nxc+1,Nyc))
        
#                #Done to preserve the boundary values, which should not enter the correction step as that is only done for u,v in the interior.
#                u = u_star
#                v = v_star
        
        #Redefines u_star and v_star as all faces excluding the boundaries, note that boundaries are still stored in u and v
        #Corrections for core
        u_star_corr = u_star[:,1:]
        v_star_corr = v_star[1:,:]
#                keyboard()
        u_star_corr_flattened = u_star_corr.flatten()
        v_star_corr_flattened = v_star_corr.flatten()
        corr_u = dt*(divxp*p_flattened)
        corr_v = dt*(divyp*p_flattened)
#                keyboard()
        u_star_corr_flattened = u_star_corr_flattened - corr_u
        v_star_corr_flattened = v_star_corr_flattened - corr_v
        
#                #Reshaping the core into a 2-D array and putting it inside the core of u and v. Note that the boundaries are already present.
#                u_star = np.reshape(u_star_flattened,(Nxc,Nyc-1))
#                v_star = np.reshape(v_star_flattened,(Nxc-1,Nyc))
        u = u_initial
        v = v_initial
        u[:,1:] = np.reshape(u_star_corr_flattened,(Nxc,Nyc))
        v[1:,:] = np.reshape(v_star_corr_flattened,(Nxc,Nyc))
#                keyboard()
        u[:,0] = u[:,-1]
#                u[:,0] = u_star[:,0]
        v[0,:] = v[-1,:]
#                v_star[-1,:] = v_star[-1,:]
        print "Time:%f" %(time_t)
        time_t += dt
        it += it
        
    e_TR_u = np.linalg.norm(u-u_final)/np.sqrt(u.size)
    e_TR_v = np.linalg.norm(v-v_final)/np.sqrt(v.size)
    p = np.reshape(p_flattened,(Nxc,Nyc))
    print "For Nx = %d, Ny = %d:" %(Nxc,Nyc)
    print "RMS error for u: %2.10f" %(e_TR_u)
    print "RMS error for v: %2.10f" %(e_TR_v)
    
    return u,v,p,e_TR_u,e_TR_v

def grid_generate(Nxc,Nyc,Lx,Ly,g_type):
    
    Nyc = int(Nyc)
    Nxc = int(Nxc)
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
    if g_type == "uniform":
        xu = xsi_u*Lx
        yv = xsi_v*Ly
        print "Uniform grid created using xu = xsi_u*Lx and yv = xsi_v*Ly"
    # non-uniform grid can be specified by redefining xsi_u and xsi_v. For e.g.
        
    if g_type == "stretched":
        alp = 1.5
        xu = (np.tanh(alp*(2.*(xsi_u - 0.5)))/np.tanh(alp) + 1.)*(Lx/2.) # can be replaced by non-uniform grid
        yv = (np.tanh(alp*(2.*(xsi_v - 0.5)))/np.tanh(alp) + 1.)*(Ly/2.) # can be replaced by non-uniform grid
        print "Stretched grid created using xu = tanh(alpha*(2.*(xsi_u - 0.5)))/tanh(alpha) + 1.)*(Lx/2.) and yv = (tanh(alpha*(2.*(xsi_v - 0.5)))/tanh(alpha) + 1.)*(Ly/2.) with alpha = 1.5"
    
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
    
    # familiarize yourself with 'flattening' options
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
    
    return Xp,Dxp,Yp,Dyp,Xu,Dxu,Yu,Dyu,Xv,Dxv,Yv,Dyv,pressureCells_Mask,u_velocityCells_Mask,v_velocityCells_Mask


def cavity_RK2SD2(Xp,Dxp,Yp,Dyp,Xu,Dxu,Yu,Dyu,Xv,Dxv,Yv,Dyv,Nxc,Nyc,Lx,Ly,pressureCells_Mask,u_velocityCells_Mask,v_velocityCells_Mask,dt,nu,plate_u,t_final):
    
    DivGradp = spatial_operators.create_DivGrad_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,plate_u)
    DivGradp[0,:] = 0.
    DivGradp[0,0] = 1.
    DivGradu,Q1 = spatial_operators.create_DivGrad_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,plate_u,"Dirichlet",1)
#    DivGradpsi_c = spatial_operators.create_DivGrad_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,plate_u,"Dirichlet")
    DivGradv = spatial_operators.create_DivGrad_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,plate_u,"Dirichlet")
#            keyboard()
    delxu = spatial_operators.create_delx_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,"Homogeneous Dirichlet")
    delyu,Q2 = spatial_operators.create_dely_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask,plate_u,"Dirichlet",1)
    delxv = spatial_operators.create_delx_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,"Homogeneous Dirichlet")
    delyv = spatial_operators.create_dely_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask,plate_u,"Dirichlet")
    divxu = spatial_operators.create_divx_operator(Dxu,Dyu,Xu,Yu,u_velocityCells_Mask)
    divyv = spatial_operators.create_divy_operator(Dxv,Dyv,Xv,Yv,v_velocityCells_Mask)
    divxp = spatial_operators.create_divx_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask)
    divyp = spatial_operators.create_divy_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask)
        
     #initial values of u,v,p
    u_initial = np.zeros(Xu[1:-1,1:-1].shape) #np.cos(Xu[1:-1,1:-1])*np.sin(Yu[1:-1,1:-1])
    v_initial = np.zeros(Xv[1:-1,1:-1].shape) #-np.cos(Xv[1:-1,1:-1])*np.sin(Yv[1:-1,1:-1])
    p_initial = np.zeros(Xp[1:-1,1:-1].shape) #-0.25*(np.cos(2*Xp[1:-1,1:-1]) + np.sin(2*Yp[1:-1,1:-1]))

    u = u_initial
    v = v_initial
    p = p_initial
    
    
    time_t = 0
    it = 1
        
       
    while time_t <= t_final:
  
  #Step 1
        
        #Finding viscous terms in prediction
        #Flattening u,v,p 2-D arrays
        
        u_flattened = u.flatten()
        v_flattened = v.flatten()
        p_flattened = p.flatten()
        
        R_viscous_u = nu*(DivGradu*u_flattened + Q1)
        R_viscous_v = nu*(DivGradv*v_flattened)
        
        
        #interpolation of v
        
        v_inter_1 = 0.5*(v[:-1,:] + v[1:,:])
        v_inter_2 = 0.5*(v_inter_1[:,:-1] + v_inter_1[:,1:])
        v_inter_2 = np.c_[np.zeros(u_initial[:,0].shape),v_inter_2,np.zeros(u_initial[:,-1].shape)]
        v_inter_2_flattened = v_inter_2.flatten()
        
        
        #interpolation of u
        
        u_inter_1 = 0.5*(u[:,:-1] + u[:,1:])
        u_inter_2 = 0.5*(u_inter_1[:-1,:] + u_inter_1[1:,:])
        u_inter_2 = np.r_[[np.zeros(v_initial[0,:].shape)],u_inter_2,[plate_u*np.ones(v_initial[-1,:].shape)]]
        u_inter_2_flattened = u_inter_2.flatten()
        
        
        #Convective term
        R_convective_u = u_flattened*(delxu*u_flattened) + v_inter_2_flattened*((delyu*u_flattened) + Q2)
        R_convective_v = u_inter_2_flattened*(delxv*v_flattened) + v_flattened*(delyv*v_flattened)
        
       
        #1st-Prediction step
        
        u_star_flattened = u_flattened + dt*(-R_convective_u + R_viscous_u)
        v_star_flattened = v_flattened + dt*(-R_convective_v + R_viscous_v)
        
   #Step 2
        
        #Re-cast u_star and v_star in 2-D array form for interpolation
        
        u_star = np.reshape(u_star_flattened,(Nxc,Nyc+1))
        v_star = np.reshape(v_star_flattened,(Nxc+1,Nyc))
        
        #Patch east-west boundaries for u and north-south boundaries for v
        u_star[:,0] = u_initial[:,0]
        u_star[:,-1] = u_initial[:,-1]
        v_star[0,:] = v_initial[0,:]
        v_star[-1,:] = v_initial[-1,:]
        
        #interpolation of v
        
        v_inter_1 = 0.5*(v_star[:-1,:] + v_star[1:,:])
        v_inter_2 = 0.5*(v_inter_1[:,:-1] + v_inter_1[:,1:])
        v_inter_2 = np.c_[np.zeros(u_initial[:,0].shape),v_inter_2,np.zeros(u_initial[:,-1].shape)]
        v_inter_2_flattened = v_inter_2.flatten()
        
        #interpolation of u
        
        u_inter_1 = 0.5*(u_star[:,:-1] + u_star[:,1:])
        u_inter_2 = 0.5*(u_inter_1[:-1,:] + u_inter_1[1:,:])
        u_inter_2 = np.r_[[np.zeros(v_initial[0,:].shape)],u_inter_2,[plate_u*np.ones(v_initial[-1,:].shape)]]
        u_inter_2_flattened = u_inter_2.flatten()
        
        #Viscous terms
        R_viscous_u_1 = nu*(DivGradu*u_star.flatten() + Q1)
        R_viscous_v_1 = nu*(DivGradv*v_star.flatten()) 
        
        #Convective terms
        R_convective_u_1 = u_star_flattened*(delxu*u_star_flattened) + v_inter_2_flattened*((delyu*u_star_flattened) + Q2)
        R_convective_v_1 = u_inter_2_flattened*(delxv*v_star_flattened) + v_star_flattened*(delyv*v_star_flattened)
        
        
        #2nd Prediction step
        u_star_flattened = u_flattened + (dt/2.0)*(-R_convective_u_1 + R_viscous_u_1 - R_convective_u + R_viscous_u)
        v_star_flattened = v_flattened + (dt/2.0)*(-R_convective_v_1 + R_viscous_v_1 - R_convective_v + R_viscous_v)
        
        u_star = np.reshape(u_star_flattened,(Nxc,Nyc+1))
        v_star = np.reshape(v_star_flattened,(Nxc+1,Nyc))
        
        #Patch east-west boundaries for u and north-south boundaries for v
        u_star[:,0] = u_initial[:,0]
        u_star[:,-1] = u_initial[:,-1]
        v_star[0,:] = v_initial[0,:]
        v_star[-1,:] = v_initial[-1,:]
        
#                keyboard()
        #Pressure-Velocity Coupling
        u_star_flattened = u_star.flatten()
        v_star_flattened = v_star.flatten()
        Div_u_star = ((divxu*u_star_flattened) + (divyv*v_star_flattened))/dt
        Div_u_star[0] = 0.
        p_flattened = spysparselinalg.spsolve(DivGradp,Div_u_star)
        p = np.reshape(p_flattened,(Nxc,Nyc))
        
        #correction step
        
        
        #Redefines u_star and v_star as all faces excluding the boundaries, note that boundaries are still stored in u and v
        #Corrections for core
        u_star_corr = u_star[:,1:-1]
        v_star_corr = v_star[1:-1,:]
        u_star_corr_flattened = u_star_corr.flatten()
        v_star_corr_flattened = v_star_corr.flatten()
        u_star_corr_flattened = u_star_corr_flattened - (dt*divxp*p_flattened)
        v_star_corr_flattened = v_star_corr_flattened - (dt*divyp*p_flattened)
        
        #Reshaping the core into a 2-D array and putting it inside the core of u and v. Note that the boundaries are already present.
        u_star_corr = np.reshape(u_star_corr_flattened,(Nxc,Nyc-1))
        v_star_corr = np.reshape(v_star_corr_flattened,(Nxc-1,Nyc))
        u_star[:,1:-1] = u_star_corr
        v_star[1:-1,:] = v_star_corr
        u = u_star
        v = v_star
        
        print "Time:%f" %(time_t)
        time_t += dt
        it += 1
        
        
    return u,v,p
        
def Vorticity_Streamfunction(u,v,Xp,Yp,Dxp,Dyp,pressureCells_Mask,plate_u):
      
    #By interpolating u,v to pressure points and generating PSI
    u_interpolated = 0.5*(u[:,:-1] + u[:,1:])
    v_interpolated = 0.5*(v[:-1,:] + v[1:,:])
    curl_cyu,Q3 = spatial_operators.create_dely_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,plate_u,"Dirichlet",1)
    curl_cxv = spatial_operators.create_delx_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,"Homogeneous Dirichlet")
    v_interpolated_flattened = v_interpolated.flatten()
    u_interpolated_flattened = u_interpolated.flatten()
    
    omega_z = ((curl_cyu*u_interpolated_flattened) + Q3 - (curl_cxv*v_interpolated_flattened))
    Omega_z = np.reshape(omega_z,(Nxc,Nyc)) 
    DivGradpsi_c = spatial_operators.create_DivGrad_operator(Dxp,Dyp,Xp,Yp,pressureCells_Mask,plate_u,"Dirichlet")
    psi = spysparselinalg.spsolve(DivGradpsi_c,omega_z)
    PSI = np.reshape(psi,(Nxc,Nyc))
    
    return Omega_z,PSI
    

    
    

def Contourplot_L(X,Y,var,figure_number,flag=0):
    
#def Vorticity_Streamfunction_2():
    fig = plt.figure(figure_number, figsize=(figwidth,figheight))
    ax   = fig.add_axes([0.15,0.15,0.6,0.6])
    
    plt.axes(ax)
    if flag == 1:
        plt.contour(X[1:-1,1:-1],Y[1:-1,1:-1],var,1000)
        
    elif flag == 0:
        plt.contourf(X[1:-1,1:-1],Y[1:-1,1:-1],var,1000)
        
    
    plt.colorbar()
    
    ax.grid('on',which='both')
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.set_xlabel(r"$x$",fontsize=textFontSize)
    ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
    plt.axis("tight")
    plt.axis("equal")
    
    plt.show()

        
        
#########################################
############### Code Starts ##############
figure_folder = "../report/"

Re100 = True

Re1000 = True

TG = True    
        
if Re100:
    
    figwidth       = 10
    figheight      = 6
    lineWidth      = 4
    textFontSize   = 14
    gcafontSize    = 30
      
      
##################################################################################################################################
      
#                       Re=100                                                                                                   #
    
    y_100 = np.loadtxt("u_vs_y_Re=100.txt",usecols=(0,))    
    u_ghia_100 = np.loadtxt("u_vs_y_Re=100.txt",usecols=(1,))
    
    x_100 = np.loadtxt("v_vs_x_Re=100.txt",usecols=(0,))    
    v_ghia_100 = np.loadtxt("v_vs_x_Re=100.txt",usecols=(1,))
    
#    #Grid 1
#    Nxc  = 64
#    Nyc  = Nxc
#    Lx   = 1.#2*np.pi
#    Ly   = 1.#2*np.pi
#    dt = 0.001
#    nu = 0.01
#    plate_u = 1.0
#    t_final = 40.0
#    g_type = "uniform"
#    print "Nxc = %d, Nyc = %d" %(Nxc,Nyc)
#    print "Grid Type:" +repr(g_type)
#    print "Time Step = %2.6f" %(dt)
#    
#    print "Generating Grid....."        
#    Xp,Dxp,Yp,Dyp,Xu,Dxu,Yu,Dyu,Xv,Dxv,Yv,Dyv,pressureCells_Mask,u_velocityCells_Mask,v_velocityCells_Mask = grid_generate(Nxc,Nyc,Lx,Ly,g_type)
#    
#    print "Starting Calculations..\n"    
#    u,v,p = cavity_RK2SD2(Xp,Dxp,Yp,Dyp,Xu,Dxu,Yu,Dyu,Xv,Dxv,Yv,Dyv,Nxc,Nyc,Lx,Ly,pressureCells_Mask,u_velocityCells_Mask,v_velocityCells_Mask,dt,nu,plate_u,t_final)
#        
#    Omega_z_64,PSI_64 = Vorticity_Streamfunction(u,v,Xp,Yp,Dxp,Dyp,pressureCells_Mask,plate_u)
#    
#    u_64u = u[:,33]
#    Y_64u = Yu[1:-1,33]
#    
#    v_64u = v[33,:]
#    X_64u = Xv[33,1:-1]
    
    
    #Can be done for non-uniform grid
    
    #Grid 4
    Nxc  = 48
    Nyc  = Nxc
    Lx   = 1.#2*np.pi
    Ly   = 1.#2*np.pi
    dt = 0.001
    nu = 0.01
    plate_u = 1.0
    t_final = 40.0
    g_type = "stretched"
    print "Nxc = %d, Nyc = %d" %(Nxc,Nyc)
    print "Grid Type:" +repr(g_type)
    print "Time Step = %2.6f" %(dt)
    
    print "Generating Grid....." 
    Xp,Dxp,Yp,Dyp,Xu,Dxu,Yu,Dyu,Xv,Dxv,Yv,Dyv,pressureCells_Mask,u_velocityCells_Mask,v_velocityCells_Mask = grid_generate(Nxc,Nyc,Lx,Ly,g_type)
    
    print "Starting Calculations..\n"
    u,v,p = cavity_RK2SD2(Xp,Dxp,Yp,Dyp,Xu,Dxu,Yu,Dyu,Xv,Dxv,Yv,Dyv,Nxc,Nyc,Lx,Ly,pressureCells_Mask,u_velocityCells_Mask,v_velocityCells_Mask,dt,nu,plate_u,t_final)
        
    Omega_z_48s_100,PSI_48s_100 = Vorticity_Streamfunction(u,v,Xp,Yp,Dxp,Dyp,pressureCells_Mask,plate_u)
    
    #Extracting u,v,omega and psi data for comarison
    i,j = np.where(Xu == 0.5)
    Y_48s_100 = Yu[i[1:-1],j[0]]
    u_48s_100 = u[:,j[0]]
    
    
    i,j = np.where(Yv == 0.5)
    X_48s_100 = Xv[i[0],j[1:-1]]
    v_48s_100 = v[i[0],:]
    
    levels = np.array([-0.1175,-0.1150,-0.1100,-0.1,-0.09,-0.07,-0.05,-0.03,-0.01,-1*10**(-4),-1*10**(-5),-1*10**(-7),-1*10**(-10)])
    figure_name = "Contour plot for streamfunction for Re=100.pdf" 
    figure_file = figure_folder + figure_name
    fig = plt.figure(figsize=(figwidth,figheight))
    ax   = fig.add_axes([0.15,0.15,0.6,0.6])
    plt.title("Streamfunction at Re=100")
    plt.contour(Xp[1:-1,1:-1],Yp[1:-1,1:-1],PSI_48s_100,levels)
    plt.colorbar()
    ax.grid('on',which='both')
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.set_xlabel(r"$x$",fontsize=textFontSize)
    ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
    plt.axis("tight")
    plt.axis("equal")
    
    print "Saving Figure:" + figure_file
    plt.savefig(figure_file)
    plt.close()
    
    levels = np.array([-3.0,-2.0,-1.0,-0.5,0,0.5,1.0,2.0,3.0,4.0,5.0])
    figure_name = "Contour plot for Vorticity for Re=100.pdf" 
    figure_file = figure_folder + figure_name
    fig = plt.figure(figsize=(figwidth,figheight))
    ax   = fig.add_axes([0.15,0.15,0.6,0.6])
    plt.title("Vorticity at Re=100")
    plt.contour(Xp[1:-1,1:-1],Yp[1:-1,1:-1],Omega_z_48s_100,levels)
    plt.colorbar()
    ax.grid('on',which='both')
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.set_xlabel(r"$x$",fontsize=textFontSize)
    ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
    plt.axis("tight")
    plt.axis("equal")
    
    print "Saving Figure:" + figure_file
    plt.savefig(figure_file)
    plt.close()
#    Contourplot_L(Xp,Yp,PSI_48s,1,1)
    
    
#################################################################################################################################################    
#                               Re=1000
if Re1000:   
    
    figwidth       = 10
    figheight      = 6
    lineWidth      = 4
    textFontSize   = 14
    gcafontSize    = 30
    
    y_1000 = np.loadtxt("u_vs_y_Re=1000.txt",usecols=(0,))    
    u_ghia_1000 = np.loadtxt("u_vs_y_Re=1000.txt",usecols=(1,))
    
    x_1000 = np.loadtxt("v_vs_x_Re=1000.txt",usecols=(0,))    
    v_ghia_1000 = np.loadtxt("v_vs_x_Re=1000.txt",usecols=(1,))
    
    #Grid 1
    Nxc  = 48
    Nyc  = Nxc
    Lx   = 1.#2*np.pi
    Ly   = 1.#2*np.pi
    dt = 0.001
    nu = 0.001
    plate_u = 1.0
    t_final = 40.0
    g_type = "stretched"
    print "Nxc = %d, Nyc = %d" %(Nxc,Nyc)
    print "Grid Type:" +repr(g_type)
    print "Time Step = %2.6f" %(dt)
    
    print "Generating Grid....."        
    Xp,Dxp,Yp,Dyp,Xu,Dxu,Yu,Dyu,Xv,Dxv,Yv,Dyv,pressureCells_Mask,u_velocityCells_Mask,v_velocityCells_Mask = grid_generate(Nxc,Nyc,Lx,Ly,g_type)
    
    print "Starting Calculations..\n"    
    u,v,p = cavity_RK2SD2(Xp,Dxp,Yp,Dyp,Xu,Dxu,Yu,Dyu,Xv,Dxv,Yv,Dyv,Nxc,Nyc,Lx,Ly,pressureCells_Mask,u_velocityCells_Mask,v_velocityCells_Mask,dt,nu,plate_u,t_final)
        
    Omega_z_48s_1000,PSI_48s_1000 = Vorticity_Streamfunction(u,v,Xp,Yp,Dxp,Dyp,pressureCells_Mask,plate_u)
    
    i,j = np.where(Xu == 0.5)
    Y_48s_1000 = Yu[i[1:-1],j[0]]
    u_48s_1000 = u[:,j[0]]
    
    
    i,j = np.where(Yv == 0.5)
    X_48s_1000 = Xv[i[0],j[1:-1]]
    v_48s_1000 = v[i[0],:]
    
#    x_omega_48s_1000 = Xp[-2,1:-1]
#    Omega_wall_1000 = Omega_z_48s_1000[-2,:]
    
    levels = np.array([-0.1,-0.09,-0.07,-0.05,-0.03,-0.01,-1*10**(-4),-1*10**(-5),-1*10**(-7),-1*10**(-10),1.0*10**(-8),1.0*10**(-7),1.0*10**(-6),1.0*10**(-5),5.0*10**(-5),1.0*10**(-4),2.5*10**(-4),5.0*10**(-4),1.0*10**(-3),1.5*10**(-3),3.0*10**(-3)])
    figure_name = "Contour plot for streamfunction for Re=1000.pdf" 
    figure_file = figure_folder + figure_name
    fig = plt.figure(figsize=(figwidth,figheight))
    ax   = fig.add_axes([0.15,0.15,0.6,0.6])
    plt.title("Streamfunction, Re=1000")
    plt.contour(Xp[1:-1,1:-1],Yp[1:-1,1:-1],PSI_48s_1000,levels)
    plt.colorbar()
    ax.grid('on',which='both')
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.set_xlabel(r"$x$",fontsize=textFontSize)
    ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
    plt.axis("tight")
    plt.axis("equal")
    
    print "Saving Figure:" + figure_file
    plt.savefig(figure_file)
    plt.close()
    
    levels = np.array([-3.0,-2.0,-1.0,-0.5,0.,0.5,1.0,2.0,3.0,4.0,5.0])
    figure_name = "Contour plot for Vorticity for Re=1000.pdf" 
    figure_file = figure_folder + figure_name
    fig = plt.figure(figsize=(figwidth,figheight))
    ax   = fig.add_axes([0.15,0.15,0.6,0.6])
    plt.title("Vorticity at Re=1000")
    plt.contour(Xp[1:-1,1:-1],Yp[1:-1,1:-1],Omega_z_48s_1000,levels)
    plt.colorbar()
    ax.grid('on',which='both')
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.set_xlabel(r"$x$",fontsize=textFontSize)
    ax.set_ylabel(r"$y$",fontsize=textFontSize,rotation=90)
    plt.axis("tight")
    plt.axis("equal")
    
    print "Saving Figure:" + figure_file
    plt.savefig(figure_file)
    plt.close()
#    Contourplot_L(Xp,Yp,PSI_48s,2,1)
   
    
#    #Grid 2
#    Nxc  = 128
#    Nyc  = Nxc
#    Lx   = 1.#2*np.pi
#    Ly   = 1.#2*np.pi
#    dt = 0.0001
#    nu = 0.001
#    plate_u = 1.0
#    t_final = 25.0
#    g_type = "uniform"
#    print "Nxc = %d, Nyc = %d" %(Nxc,Nyc)
#    print "Grid Type:" +repr(g_type)
#    print "Time Step = %2.6f" %(dt)
#    
#    print "Generating Grid....." 
#    Xp,Dxp,Yp,Dyp,Xu,Dxu,Yu,Dyu,Xv,Dxv,Yv,Dyv,pressureCells_Mask,u_velocityCells_Mask,v_velocityCells_Mask = grid_generate(Nxc,Nyc,Lx,Ly,g_type)
#    
#    print "Starting Calculations..\n"
#    u,v,p = cavity_RK2SD2(Xp,Dxp,Yp,Dyp,Xu,Dxu,Yu,Dyu,Xv,Dxv,Yv,Dyv,Nxc,Nyc,Lx,Ly,pressureCells_Mask,u_velocityCells_Mask,v_velocityCells_Mask,dt,nu,plate_u,t_final)
#        
#    Omega_z_128,PSI_128 = Vorticity_Streamfunction(u,v,Xp,Yp,Dxp,Dyp,pressureCells_Mask,plate_u)
#    
#    u_128u = u[:,65]
#    Y_128u = Yu[1:-1,65]
#    
#    v_128u = v[65,:]
#    X_128u = Xv[65,1:-1]
    
figure_name = "Variation of u with y at geometric centre.pdf"
figure_file = figure_folder + figure_name       
fig = plt.figure(figsize=(figwidth,figheight))
plt.grid('on',which='both')
plt.xlabel("y",fontsize=textFontSize)
plt.ylabel(r"u",fontsize=textFontSize,rotation=90)
plt.title("u vs y at x=0.5")
if Re100:
    plt.plot(Y_48s_100,u_48s_100,'k-',label="Re=100: 48x48 stretched")
    plt.plot(y_100,u_ghia_100,'r*',label="Re=100:Ghia et al.")
if Re1000:
    plt.plot(Y_48s_1000,u_48s_1000,'r-',label="Re=1000: 48x48 stretched")
    plt.plot(y_1000,u_ghia_1000,'k+',label="Re=1000:Ghia et al.")

plt.legend(loc='best',numpoints=1)

print "Saving Figure:" + figure_file
plt.savefig(figure_file)
plt.close()
#    


figure_name = "Variation of v with x at geometric centre.pdf"
figure_file = figure_folder + figure_name       
fig = plt.figure(figsize=(figwidth,figheight))
plt.grid('on',which='both')
plt.xlabel("y",fontsize=textFontSize)
plt.ylabel(r"u",fontsize=textFontSize,rotation=90)
plt.title("v vs x at y=0.5")
if Re100:
    plt.plot(X_48s_100,v_48s_100,'k-',label="Re=100: 48x48 stretched")
    plt.plot(x_100,v_ghia_100,'k*',label="Re=100:Ghia et al.")
if Re1000:
    plt.plot(X_48s_1000,v_48s_1000,'r-',label="Re=1000: 48x48 stretched")
    plt.plot(x_1000,v_ghia_1000,'k+',label="Re=1000:Ghia et al.")

plt.legend(loc='best',numpoints=1)

print "Saving Figure:" + figure_file
plt.savefig(figure_file)
plt.close()
    
    
#    Contourplot(Xp,Yp,PSI,0)
#    
#    Contourplot(Xp,Yp,p,1)
#    
#    Contourplot(Xu,Yu,u,2)
#    
#    Contourplot(Xv,Yv,v,3)
#    
#    Contourplot(Xp,Yp,Omega_z,4,1)
    
if TG:
    
    figwidth       = 10
    figheight      = 6
    lineWidth      = 4
    textFontSize   = 14
    gcafontSize    = 30
    NXC  = np.array([10,25,50,80])
    NYC  = NXC
    Lx   = 2.*np.pi
    Ly   = 2.*np.pi
    t_final = 1.0
    dt = 0.0001
    
    TR_u_RK1 = np.zeros(NXC.size)
    TR_v_RK1 = np.zeros(NXC.size)
    TR_u_RK2 = np.zeros(NXC.size)
    TR_v_RK2 = np.zeros(NXC.size)
    TR_u_RK3 = np.zeros(NXC.size)
    TR_v_RK3 = np.zeros(NXC.size)
    TR_u_RK4 = np.zeros(NXC.size)
    TR_v_RK4 = np.zeros(NXC.size)
    
    for i,Nxc in enumerate(NXC):
        Nxc = int(Nxc)
        Nyc = int(NYC[i])
        Xp,Dxp,Yp,Dyp,Xu,Dxu,Yu,Dyu,Xv,Dxv,Yv,Dyv,pressureCells_Mask,u_velocityCells_Mask,v_velocityCells_Mask = Grid_Generate_TG(Nxc,Nyc,Lx,Ly)
        u1,v1,p1,TR_u_RK1[i],TR_v_RK1[i] = SD2RK1_TG(Nxc,Nyc,Lx,Ly,t_final,dt,Xp,Dxp,Yp,Dyp,Xu,Dxu,Yu,Dyu,Xv,Dxv,Yv,Dyv,pressureCells_Mask,u_velocityCells_Mask,v_velocityCells_Mask)
        u2,v2,p2,TR_u_RK2[i],TR_v_RK2[i] = SD2RK2_TG(Nxc,Nyc,Lx,Ly,t_final,dt,Xp,Dxp,Yp,Dyp,Xu,Dxu,Yu,Dyu,Xv,Dxv,Yv,Dyv,pressureCells_Mask,u_velocityCells_Mask,v_velocityCells_Mask)
        u3,v3,p3,TR_u_RK3[i],TR_v_RK3[i] = SD2RK3_TG(Nxc,Nyc,Lx,Ly,t_final,dt,Xp,Dxp,Yp,Dyp,Xu,Dxu,Yu,Dyu,Xv,Dxv,Yv,Dyv,pressureCells_Mask,u_velocityCells_Mask,v_velocityCells_Mask)
        u4,v4,p4,TR_u_RK4[i],TR_v_RK4[i] = SD2RK4_TG(Nxc,Nyc,Lx,Ly,t_final,dt,Xp,Dxp,Yp,Dyp,Xu,Dxu,Yu,Dyu,Xv,Dxv,Yv,Dyv,pressureCells_Mask,u_velocityCells_Mask,v_velocityCells_Mask)
        
        
    figure_name = "RMS error of u vs Grid spacing for Taylor Green Vortex.pdf"
    fig = plt.figure(1, figsize=(figwidth,figheight))
    plt.title("$epsilon_{RMS}$ vs $\Delta x$")
    plt.xlabel(r"Grid spacing $\Delta x$",fontsize=textFontSize)
    plt.ylabel(r"$epsilon_{RMS}$",fontsize=textFontSize,rotation=90)
    plt.loglog(2*np.pi/NXC,TR_u_RK1,'b--',label="RK1")
    plt.loglog(2*np.pi/NXC,TR_u_RK2,'r--',label="RK2")
    plt.loglog(2*np.pi/NXC,TR_u_RK3,'k--',label="RK3")
    plt.loglog(2*np.pi/NXC,TR_u_RK4,'g--',label="RK4")
    plt.loglog(2*np.pi/NXC,(2*np.pi/NXC)**2,'k-',label="Order 2")
    plt.loglog(2*np.pi/NXC,(2*np.pi/NXC)**3,'b-',label="Order 3")
    plt.loglog(2*np.pi/NXC,(2*np.pi/NXC)**1,'r-',label="Order 1")
    plt.legend(loc='best')
    figure_file_path = figure_folder + figure_name
    print "Saving figure: " + figure_file_path
    plt.tight_layout()
    plt.savefig(figure_file_path)
    plt.close()
    
    figure_name = "RMS error of v vs Grid spacing for Taylor Green Vortex.pdf"
    fig = plt.figure(2, figsize=(figwidth,figheight))
    plt.title("$epsilon_{RMS}$ vs $\Delta x$")
    plt.xlabel(r"Grid spacing $\Delta x$",fontsize=textFontSize)
    plt.ylabel(r"$epsilon_{RMS}$",fontsize=textFontSize,rotation=90)
    plt.loglog(2*np.pi/NXC,TR_v_RK1,'b--',label="RK1")
    plt.loglog(2*np.pi/NXC,TR_v_RK2,'r--',label="RK2")
    plt.loglog(2*np.pi/NXC,TR_v_RK3,'k--',label="RK3")
    plt.loglog(2*np.pi/NXC,TR_v_RK4,'g--',label="RK4")
    plt.loglog(2*np.pi/NXC,(2*np.pi/NXC)**2,'k-',label="Order 2")
    plt.loglog(2*np.pi/NXC,(2*np.pi/NXC)**3,'b-',label="Order 3")
    plt.loglog(2*np.pi/NXC,(2*np.pi/NXC)**1,'r-',label="Order 1")
    plt.legend(loc='best')
    figure_file_path = figure_folder + figure_name
    print "Saving figure: " + figure_file_path
    plt.tight_layout()
    plt.savefig(figure_file_path)
    plt.close()
        
    
    
