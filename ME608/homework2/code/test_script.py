
import os
import sys
from pdb import set_trace
import scipy as sp
import numpy as np
import scipy.sparse as scysparse
import scipy.sparse.linalg as splinalg
import pylab as plt
import plot_data
import bivariate_fit as fit

def test(N,mesh_no):
    
    figure_folder = "../report/figures/"
    # N = total number of points surrouding a centroid AND including the centroid itself
    mesh_abs_tol = 10**(-8)
    xc,yc    = 0.,0. # centroid location
    
    # Analytical functions
    phi       = lambda X,Y: np.cos(2*X-2*Y-np.pi/4.)+X+Y
    dphi_dx   = lambda X,Y: 1. - 2.*np.sin(2*X - 2*Y - np.pi/4.)
    dphi_dy   = lambda X,Y: 1. - 2.*np.sin(2*Y - 2*X + np.pi/4.)
    d2phi_dx2 = lambda X,Y: -4.*np.cos(2*X - 2*Y - np.pi/4.)
    d2phi_dy2 = lambda X,Y: -4.*np.cos(2*Y - 2*X + np.pi/4.)
    
    # Exact values
    phi_exact      = phi(xc,yc)
    dphidx_exact   = dphi_dx(xc,yc)
    dphidy_exact   = dphi_dy(xc,yc)
    d2phidx2_exact = d2phi_dx2(xc,yc)
    d2phidy2_exact = d2phi_dy2(xc,yc)
    
    error = {'phi_x':[],'phi_y':[],'dphi_dx':[],'dphi_dy':[],'d2phi_dx2':[],'d2phi_dy2':[]}
    mesh_size = np.logspace(1.,-5,100) # 
    for delta_x in mesh_size: 
      delta_y  = np.abs(dphidx_exact/dphidy_exact*delta_x)
    
    ## Around a Circle
      theta = np.linspace(0.,2*np.pi,N-1,endpoint=False)
      theta_off = 0.*np.pi/3.
      xn,yn    = delta_x*np.cos(theta+theta_off),delta_y*np.sin(theta+theta_off)
    
    ## Manual Input
#      xn   = np.array([-3.0/2.0*delta_x,-delta_x/2.0,delta_x/2.0,+3.0/2.0*delta_x, -delta_x/2.0,delta_x/2.0,-delta_x/2.0,delta_x/2.0])
#      yn   = np.array([0.,  0.            ,0.         ,0., delta_y,delta_y,-delta_y,-delta_y])
                      
      n_nodes = len(xn)
      xc_eff = xc + np.random.randn()*mesh_abs_tol #*delta_x
      yc_eff = yc + np.random.randn()*mesh_abs_tol #*delta_y
      xn     += np.random.randn(n_nodes)*mesh_abs_tol
      yn     += np.random.randn(n_nodes)*mesh_abs_tol
      phi_num_x,dphidx_num,d2phidx2_num = fit.BiVarPolyFit_X(xc_eff,yc_eff,xn,yn,phi(xn,yn))
      phi_num_y,dphidy_num,d2phidy2_num = fit.BiVarPolyFit_Y(xc_eff,yc_eff,xn,yn,phi(xn,yn))
     
      if delta_x==mesh_size[-1]:
       weights_dx = np.zeros(n_nodes)
       weights_dy = np.zeros(n_nodes)
       weights_dx2 = np.zeros(n_nodes)
       weights_dy2 = np.zeros(n_nodes)
       for ino in range(0,n_nodes):
          phi_base = np.zeros(n_nodes)
          phi_base[ino] = 1.0
          _,weights_dx[ino],weights_dx2[ino] = fit.BiVarPolyFit_X(xc_eff,yc_eff,xn,yn,phi_base)
          _,weights_dy[ino],weights_dy2[ino] = fit.BiVarPolyFit_Y(xc_eff,yc_eff,xn,yn,phi_base)
      
      error['phi_x'].append(np.abs(phi_exact - phi_num_x))
      error['phi_y'].append(np.abs(phi_exact - phi_num_y))
      error['dphi_dx'].append(np.abs(dphidx_exact - dphidx_num))
      error['dphi_dy'].append(np.abs(dphidy_exact - dphidy_num))
      error['d2phi_dx2'].append(np.abs(d2phidx2_exact - d2phidx2_num))
      error['d2phi_dy2'].append(np.abs(d2phidy2_exact - d2phidy2_num))
    
    #set_trace()
    
    #############################################
    ############### Plotting ####################
    #############################################
    
    # LaTeX preamble
    from matplotlib import rc as matplotlibrc
    matplotlibrc('text.latex', preamble='\usepackage{color}')
    matplotlibrc('text',usetex=True)
    matplotlibrc('font', family='serif')
    
    # Reference decay rates
    ref_error_2nd = mesh_size*mesh_size
    ref_error_2nd /= ref_error_2nd[0]
    ref_error_1nd = mesh_size
    ref_error_1nd /= ref_error_1nd[0]
    inv_mesh_size = 1./mesh_size
    
    ########################################
    figwidth,figheight = 8,8
    lineWidth = 3
    fontSize = 25
    gcafontSize = 21
    
    fig = plt.figure(0,(figwidth,figheight))
    ax = fig.add_subplot(1,1,1)
    ax.plot(xn/delta_x,yn/delta_y,'ko',markersize = 12)
    ax.plot(xc/delta_x,yc/delta_y,'x',markersize = 12)
    ax.set_xlabel(r"$1/h$",fontsize=fontSize)
    ax.set_xlabel(r"$x/\Delta x$",fontsize=fontSize)
    ax.set_ylabel(r"$y/\Delta y$",fontsize=fontSize)
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('Points_Arragement.pdf')
    plt.close()
    
    ########################################
    figwidth,figheight = 14,12
    fig = plt.figure(1,(figwidth,figheight))
    ax = fig.add_subplot(2,3,1)
    error_plot = error['phi_x']
    ax.loglog(inv_mesh_size,error_plot,linewidth=lineWidth)
    ax.loglog(inv_mesh_size,ref_error_1nd*error_plot[0],':k')
    ax.loglog(inv_mesh_size,1e-1*ref_error_2nd*error_plot[0],':k')
    ax.set_title(r"X direction interpolation",fontsize=gcafontSize)
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.set_xlabel(r"$1/h$",fontsize=fontSize)
    ax.set_ylabel(r"error",fontsize=fontSize)
    ax.grid(True)
    
    ax = fig.add_subplot(2,3,2)
    error_plot = error['dphi_dx']
    ax.loglog(inv_mesh_size,error_plot,linewidth=lineWidth)
    ax.loglog(inv_mesh_size,ref_error_1nd*error_plot[0],':k')
    ax.loglog(inv_mesh_size,1e-1*ref_error_2nd*error_plot[0],':k')
    ax.set_title(r"dphi/dx",fontsize=gcafontSize)
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.set_xlabel(r"$1/h$",fontsize=fontSize)
    ax.grid(True)
    
    ax = fig.add_subplot(2,3,3)
    error_plot = error['d2phi_dx2']
    ax.loglog(inv_mesh_size,error_plot,linewidth=lineWidth)
    ax.loglog(inv_mesh_size,ref_error_1nd*error_plot[0],':k')
    ax.loglog(inv_mesh_size,1e-1*ref_error_2nd*error_plot[0],':k')
    ax.set_title(r"d2phi/dx2",fontsize=gcafontSize)
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.set_xlabel(r"$1/h$",fontsize=fontSize)
    ax.grid(True)
    
    ax = fig.add_subplot(2,3,4)
    error_plot = error['phi_y']
    ax.loglog(inv_mesh_size,error_plot,linewidth=lineWidth)
    ax.loglog(inv_mesh_size,ref_error_1nd*error_plot[0],':k')
    ax.loglog(inv_mesh_size,1e-1*ref_error_2nd*error_plot[0],':k')
    ax.set_title(r"Y direction interpolation",fontsize=gcafontSize)
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.set_xlabel(r"$1/h$",fontsize=fontSize)
    ax.set_ylabel(r"error",fontsize=fontSize)
    ax.grid(True)
    
    ax = fig.add_subplot(2,3,5)
    error_plot = error['dphi_dy']
    ax.loglog(inv_mesh_size,error_plot,linewidth=lineWidth)
    ax.loglog(inv_mesh_size,ref_error_1nd*error_plot[0],':k')
    ax.loglog(inv_mesh_size,1e-1*ref_error_2nd*error_plot[0],':k')
    ax.set_title(r"dphi/dy",fontsize=gcafontSize)
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.set_xlabel(r"$1/h$",fontsize=fontSize)
    ax.grid(True)
    
    ax = fig.add_subplot(2,3,6)
    error_plot = error['d2phi_dy2']
    ax.loglog(inv_mesh_size,error_plot,linewidth=lineWidth)
    ax.loglog(inv_mesh_size,ref_error_1nd*error_plot[0],':k')
    ax.loglog(inv_mesh_size,1e-1*ref_error_2nd*error_plot[0],':k')
    ax.set_title(r"d2phi/dy2",fontsize=gcafontSize)
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.set_xlabel(r"$1/h$",fontsize=fontSize)
    ax.grid(True)
    
    fig_name = figure_folder + 'Mesh ' + str(mesh_no) + ', Error_vs_invMeshSize (N=' + str(N) + ').pdf' 
    plt.tight_layout()
    plt.show()
    plt.savefig(fig_name)
    plt.close()
