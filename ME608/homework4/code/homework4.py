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
#import bivariate_fit as fit
import umesh_reader
#import plot_data
from scipy.interpolate import griddata
from timeit import default_timer
import Operators

plt.close('all')

figure_folder = "../report/figures/"
icemcfd_project_folder = './mesh/'
filename = 'Lid.msh'
mshfile_fullpath = icemcfd_project_folder + filename

lid_v = 1.
nu = 0.01
dt = 0.01
t_final = 20.0

figwidth,figheight = 14,12
lineWidth = 3
fontSize = 25
gcafontSize = 21
textFontSize = 14

p1 = True
#p1RK2 = True
p2 = True
p3 = True


if p1:
    
    print "Part a"
    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)
    
    ####################################################
    ######### Plot Grid Labels / Connectivity ##########
    ####################################################
    
    fig_width = 30
    fig_height = 17
    textFontSize   = 15
    gcafontSize    = 32
    lineWidth      = 2
    
    Plot_Node_Labels = False
    Plot_Face_Labels = False
    Plot_CV_Labels   = False
    
    # the following enables LaTeX typesetting, which will cause the plotting to take forever..
    # from matplotlib import rc as matplotlibrc
    # matplotlibrc('text.latex', preamble='\usepackage{color}')
    # matplotlibrc('text',usetex=True)
    # matplotlibrc('font', family='serif')
    
    mgplx = 0.05*np.abs(max(xy_no[:,0])-min(xy_no[:,0]))
    mgply = 0.05*np.abs(max(xy_no[:,1])-min(xy_no[:,1]))
    xlimits = [min(xy_no[:,0])-mgplx,max(xy_no[:,0])+mgplx]
    ylimits = [min(xy_no[:,1])-mgply,max(xy_no[:,1])+mgply]
    
    fig = plt.figure(0,figsize=(fig_width,fig_height))
    ax = fig.add_subplot(111)
    ax.plot(xy_no[:,0],xy_no[:,1],'o',markersize=5,markerfacecolor='k')
    
    node_color = 'k'
    centroid_color = 'r'
    
    for inos_of_fa in noofa:
       ax.plot(xy_no[inos_of_fa,0], xy_no[inos_of_fa,1], 'k-', linewidth = lineWidth)
    
    if Plot_Face_Labels:
      nfa = xy_fa.shape[0] # number of faces
      faces_indexes = range(0,nfa)
      for x_fa,y_fa,ifa in zip(xy_fa[:,0],xy_fa[:,1],faces_indexes):
        ax.text(x_fa,y_fa,repr(ifa),transform=ax.transData,color='k',
            verticalalignment='center',horizontalalignment='center',fontsize=textFontSize )
    
    if Plot_Node_Labels:
      nno = xy_no.shape[0] # number of nodes
      node_indexes = range(0,nno)
      for xn,yn,ino in zip(xy_no[:,0],xy_no[:,1],node_indexes):
        ax.text(xn,yn,repr(ino),transform=ax.transData,color='r',
            verticalalignment='top',horizontalalignment='left',fontsize=textFontSize )
    
    if Plot_CV_Labels:
      ncv = xy_cv.shape[0]  # number of control volumes
      cv_indexes = range(0,ncv)
      for xcv,ycv,icv in zip(xy_cv[:,0],xy_cv[:,1],cv_indexes):
        ax.text(xcv,ycv,repr(icv),transform=ax.transData,color='b',
            verticalalignment='top',horizontalalignment='left',fontsize=textFontSize )
    
    ax.axis('equal')
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    ax.set_xlabel(r'$x$',fontsize=1.5*gcafontSize)
    ax.set_ylabel(r'$y$',fontsize=1.5*gcafontSize)
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
#    fig_name = filename.split('.')[0]+'.pdf'
    figure_name = "Mesh.pdf"
    figure_file = figure_folder + figure_name
    plt.savefig(figure_file)
    plt.close()
#    #
#    keyboard()
    
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    faono1 = np.array(faono) # Converting faono to an array
    wall = np.where(partofa1=='WALL')   # Vectorized approach to find face belonging to part 'COLD'
    lid = np.where(partofa1=='LID')   # Vectorized approach to find face belonging to part 'HOT'
    solid = np.where(partofa1=='SOLID')   # Vectorized approach to find face belonging to part 'SOLID'
    
    
    nno_int = nno-np.unique(noofa[lid]).size-np.unique(noofa[wall]).size # No. of internal nodes
    nfa_int = np.size(np.where(partofa1=='SOLID'))
    
    u = np.zeros(nno)
    v = np.zeros(nno)
    
    u_temp = np.zeros(nno)
    v_temp = np.zeros(nno)
    
    p = np.zeros(ncv)
    
    u[np.unique(noofa[lid])] = lid_v
    
    delx, dely = Operators.gradient_nn(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa)
    
    Divx, Divy = Operators.Divergence_facv(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa)
    
    Gradx, Grady = Operators.Gradient_cvfa(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa1)
    
    DivGrad = (Divx*Gradx) + (Divy*Grady)
    
    DivGrad[0,:] = 0.
    DivGrad[0,0] = 1.
    
    Avg_n2fa = Operators.Avg_nfa_int(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa)
    
    Avgfa2n = Operators.Avg_fan_int(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa)
       
    laplace = (delx*delx) + (dely*dely)
    laplace[nno_int:,:] = 0.
    
    delx[nno_int:,:] = 0.
    dely[nno_int:,:] = 0.
    
    t = 0.
    it = 1
    
    while t<t_final:
        
        u_temp = u
        v_temp = v
    
        u_fa = np.zeros(nfa)
        v_fa = np.zeros(nfa)
        
        u_star = np.zeros(nno)
        v_star = np.zeros(nno)
        
        u_star_fa = np.zeros(nfa)
        v_star_fa = np.zeros(nfa)
        
        
        R_viscous_u = nu*(laplace*u)
        R_viscous_v = nu*(laplace*v)
        
        R_convective_u = (u*(delx*u)) + (v*(dely*u))
        R_convective_v = (u*(delx*v)) + (v*(dely*v))
        
        u_star = u  + (dt*(-R_convective_u + R_viscous_u))
        v_star = v  + (dt*(-R_convective_v + R_viscous_v))
        
        u_star_fa = Avg_n2fa*u_star
        v_star_fa = Avg_n2fa*v_star
        
        #    for i in np.arange(nfa_int):
        #        nodes = noofa[i]
        #        u_star_fa[i] = (u_star[nodes[0]] + u_star[nodes[1]])/2.
        #        v_star_fa[i] = (v_star[nodes[0]] + v_star[nodes[1]])/2.
        #        
        u_star_fa[lid] = lid_v
        v_star_fa[lid] = 0.
        
        u_star_fa[wall] = 0.
        v_star_fa[wall] = 0.
        
        #    keyboard()
        
        q = ((Divx*u_star_fa) + (Divy*v_star_fa))/dt
        q[0] = 0.
        
        p = splinalg.spsolve(DivGrad,q)
        
        u_fa = u_star_fa - (dt*(Gradx*p))
        
        v_fa = v_star_fa - (dt*(Grady*p))
        
        u = Avgfa2n*u_fa
        v = Avgfa2n*v_fa
        
        u[np.unique(noofa[lid])] = lid_v
        v[np.unique(noofa[lid])] = 0.
        
#        if np.isnan(np.average(u))==True | np.isnan(np.average(v))==True:
#            
#            u = u_temp
#            v = v_temp
#            break
        
#        for i in np.arange(nno_int):
#            u[i] = np.average(u_fa[np.unique(faono1[i])])
#            v[i] = np.average(v_fa[np.unique(faono1[i])])
        
        print "Time %f" %(t)    
        t += dt
        print "Iteration %d" %(it)
        
        it += 1
        
        print "u_average:"
        print np.average(u)
        
        print "v_average:"
        print np.average(v)
     
    y_100 = np.loadtxt("u_vs_y_Re=100.txt",usecols=(0,))    
    u_ghia_100 = np.loadtxt("u_vs_y_Re=100.txt",usecols=(1,))
    
    x_100 = np.loadtxt("v_vs_x_Re=100.txt",usecols=(0,))    
    v_ghia_100 = np.loadtxt("v_vs_x_Re=100.txt",usecols=(1,))
    
    n = 100
    xu = 0.5*np.ones(n)
    yu = np.linspace(xy_no[:,1].min(),xy_no[:,1].max(),n)
    #XU,YU = np.meshgrid(xu,yu)
    # interpolate Z values on defined grid
    U = griddata(np.vstack((xy_no[:,0].flatten(),xy_no[:,1].flatten())).T, np.vstack(u.flatten()),(xu,yu),method="cubic")
    
    yv = 0.5*np.ones(n)
    xv = np.linspace(xy_no[:,0].min(),xy_no[:,0].max(),n)
    XV,YV = np.meshgrid(xv,yv)
    
    V = griddata(np.vstack((xy_no[:,0].flatten(),xy_no[:,1].flatten())).T, np.vstack(v.flatten()),(xv,yv),method="cubic")
    
    figure_name = "M1:Variation of u with y at geometric centre.pdf"
    figure_file = figure_folder + figure_name       
    fig = plt.figure(figsize=(figwidth,figheight))
    plt.grid('on',which='both')
    plt.xlabel("y",fontsize=textFontSize)
    plt.ylabel(r"u",fontsize=textFontSize,rotation=90)
    plt.title("u vs y at x=0.5")
    plt.plot(yu,U,'k-',label="Re=100: Simulation")
    plt.plot(y_100,u_ghia_100,'r*',label="Re=100:Ghia et al.")
    plt.savefig(figure_file)
    plt.close()
    
    figure_name = "M1:Variation of v with x at geometric centre.pdf"
    figure_file = figure_folder + figure_name       
    fig = plt.figure(figsize=(figwidth,figheight))
    plt.grid('on',which='both')
    plt.xlabel("x",fontsize=textFontSize)
    plt.ylabel(r"v",fontsize=textFontSize,rotation=90)
    plt.title("v vs x at x=0.5")
    plt.plot(xv,V,'k-',label="Re=100: Simulation")
    plt.plot(x_100,v_ghia_100,'r*',label="Re=100:Ghia et al.")
    plt.savefig(figure_file)
    plt.close()
    
    print ".............................................................................."
    
    #plot_data.plot_data(xy_no[:,0],xy_no[:,1],v,"temp")
    #plot_data.plot_data(xy_no[:,0],xy_no[:,1],u,"temp1")
    
#if p1RK2:  
#    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)
#    
##    ####################################################
##    ######### Plot Grid Labels / Connectivity ##########
##    ####################################################
##    
##    fig_width = 30
##    fig_height = 17
##    textFontSize   = 15
##    gcafontSize    = 32
##    lineWidth      = 2
##    
##    Plot_Node_Labels = False
##    Plot_Face_Labels = False
##    Plot_CV_Labels   = False
##    
##    # the following enables LaTeX typesetting, which will cause the plotting to take forever..
##    # from matplotlib import rc as matplotlibrc
##    # matplotlibrc('text.latex', preamble='\usepackage{color}')
##    # matplotlibrc('text',usetex=True)
##    # matplotlibrc('font', family='serif')
##    
##    mgplx = 0.05*np.abs(max(xy_no[:,0])-min(xy_no[:,0]))
##    mgply = 0.05*np.abs(max(xy_no[:,1])-min(xy_no[:,1]))
##    xlimits = [min(xy_no[:,0])-mgplx,max(xy_no[:,0])+mgplx]
##    ylimits = [min(xy_no[:,1])-mgply,max(xy_no[:,1])+mgply]
##    
##    fig = plt.figure(0,figsize=(fig_width,fig_height))
##    ax = fig.add_subplot(111)
##    ax.plot(xy_no[:,0],xy_no[:,1],'o',markersize=5,markerfacecolor='k')
##    
##    node_color = 'k'
##    centroid_color = 'r'
##    
##    for inos_of_fa in noofa:
##       ax.plot(xy_no[inos_of_fa,0], xy_no[inos_of_fa,1], 'k-', linewidth = lineWidth)
##    
##    if Plot_Face_Labels:
##      nfa = xy_fa.shape[0] # number of faces
##      faces_indexes = range(0,nfa)
##      for x_fa,y_fa,ifa in zip(xy_fa[:,0],xy_fa[:,1],faces_indexes):
##        ax.text(x_fa,y_fa,repr(ifa),transform=ax.transData,color='k',
##            verticalalignment='center',horizontalalignment='center',fontsize=textFontSize )
##    
##    if Plot_Node_Labels:
##      nno = xy_no.shape[0] # number of nodes
##      node_indexes = range(0,nno)
##      for xn,yn,ino in zip(xy_no[:,0],xy_no[:,1],node_indexes):
##        ax.text(xn,yn,repr(ino),transform=ax.transData,color='r',
##            verticalalignment='top',horizontalalignment='left',fontsize=textFontSize )
##    
##    if Plot_CV_Labels:
##      ncv = xy_cv.shape[0]  # number of control volumes
##      cv_indexes = range(0,ncv)
##      for xcv,ycv,icv in zip(xy_cv[:,0],xy_cv[:,1],cv_indexes):
##        ax.text(xcv,ycv,repr(icv),transform=ax.transData,color='b',
##            verticalalignment='top',horizontalalignment='left',fontsize=textFontSize )
##    
##    ax.axis('equal')
##    ax.set_xlim(xlimits)
##    ax.set_ylim(ylimits)
##    ax.set_xlabel(r'$x$',fontsize=1.5*gcafontSize)
##    ax.set_ylabel(r'$y$',fontsize=1.5*gcafontSize)
##    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
##    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
###    fig_name = filename.split('.')[0]+'.pdf'
##    figure_name = "Mesh.pdf"
##    figure_file = figure_folder + figure_name
##    plt.savefig(figure_file)
##    plt.close()
##    #
##    keyboard()
#    
#    nno = xy_no.shape[0]  # No. of nodes
#    ncv = xy_cv.shape[0]  # No. of CVs
#    nfa = xy_fa.shape[0]  # No. of faces
#    partofa1 = np.array(partofa) # Converting partofa to an array
#    faono1 = np.array(faono) # Converting faono to an array
#    wall = np.where(partofa1=='WALL')   # Vectorized approach to find face belonging to part 'COLD'
#    lid = np.where(partofa1=='LID')   # Vectorized approach to find face belonging to part 'HOT'
#    solid = np.where(partofa1=='SOLID')   # Vectorized approach to find face belonging to part 'SOLID'
#    
#    
#    nno_int = nno-np.unique(noofa[lid]).size-np.unique(noofa[wall]).size # No. of internal nodes
#    nfa_int = np.size(np.where(partofa1=='SOLID'))
#    
#    u = np.zeros(nno)
#    v = np.zeros(nno)
#    
#    u_temp = np.zeros(nno)
#    v_temp = np.zeros(nno)
#    
#    p = np.zeros(ncv)
#    
#    u[np.unique(noofa[lid])] = lid_v
#    
#    delx, dely = Operators.gradient_nn(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa)
#    
#    Divx, Divy = Operators.Divergence_facv(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa)
#    
#    Gradx, Grady = Operators.Gradient_cvfa(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa1)
#    
#    DivGrad = (Divx*Gradx) + (Divy*Grady)
#    
#    DivGrad[0,:] = 0.
#    DivGrad[0,0] = 1.
#    
#    Avg_n2fa = Operators.Avg_nfa_int(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa)
#    
#    Avgfa2n = Operators.Avg_fan_int(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa)
#       
#    laplace = (delx*delx) + (dely*dely)
#    laplace[nno_int:,:] = 0.
#    
#    delx[nno_int:,:] = 0.
#    dely[nno_int:,:] = 0.
#    
#    t = 0.
#    it = 1
#    
#    while t<t_final:
#        
#        u_temp = u
#        v_temp = v
#    
#        u_fa = np.zeros(nfa)
#        v_fa = np.zeros(nfa)
#        
#        u_star = np.zeros(nno)
#        v_star = np.zeros(nno)
#        
#        u_star_fa = np.zeros(nfa)
#        v_star_fa = np.zeros(nfa)
#        
#        
#        R_viscous_u = nu*(laplace*u)
#        R_viscous_v = nu*(laplace*v)
#        
#        R_convective_u = (u*(delx*u)) + (v*(dely*u))
#        R_convective_v = (u*(delx*v)) + (v*(dely*v))
#        
#        u_star = u  + (dt*(-R_convective_u + R_viscous_u))
#        v_star = v  + (dt*(-R_convective_v + R_viscous_v))
#        
#        R_viscous_u_1 = nu*(laplace*u_star)
#        R_viscous_v_1 = nu*(laplace*v_star)
#        
#        R_convective_u_1 = (u_star*(delx*u_star)) + (v_star*(dely*u_star))
#        R_convective_v_1 = (u_star*(delx*v_star)) + (v_star*(dely*v_star))
#        
#        u_star = u  + ((dt*(-R_convective_u -R_convective_u_1 + R_viscous_u + R_viscous_u_1))/2.)
#        v_star = v  + ((dt*(-R_convective_v -R_convective_v_1 + R_viscous_v + R_viscous_v_1))/2.)
#        
#        u_star_fa = Avg_n2fa*u_star
#        v_star_fa = Avg_n2fa*v_star
#        
#        #    for i in np.arange(nfa_int):
#        #        nodes = noofa[i]
#        #        u_star_fa[i] = (u_star[nodes[0]] + u_star[nodes[1]])/2.
#        #        v_star_fa[i] = (v_star[nodes[0]] + v_star[nodes[1]])/2.
#        #        
#        u_star_fa[lid] = lid_v
#        v_star_fa[lid] = 0.
#        
#        u_star_fa[wall] = 0.
#        v_star_fa[wall] = 0.
#        
#        #    keyboard()
#        
#        q = ((Divx*u_star_fa) + (Divy*v_star_fa))/dt
#        q[0] = 0.
#        
#        p = splinalg.spsolve(DivGrad,q)
#        
#        u_fa = u_star_fa - (dt*(Gradx*p))
#        
#        v_fa = v_star_fa - (dt*(Grady*p))
#        
#        u = Avgfa2n*u_fa
#        v = Avgfa2n*v_fa
#        
#        u[np.unique(noofa[lid])] = lid_v
#        v[np.unique(noofa[lid])] = 0.
#        
##        if np.isnan(np.average(u))==True | np.isnan(np.average(v))==True:
##            
##            u = u_temp
##            v = v_temp
##            break
#        
##        for i in np.arange(nno_int):
##            u[i] = np.average(u_fa[np.unique(faono1[i])])
##            v[i] = np.average(v_fa[np.unique(faono1[i])])
#        
#        print "Time %f" %(t)    
#        t += dt
#        print "Iteration %d" %(it)
#        
#        it += 1
#        
#        print "u_average:"
#        print np.average(u)
#        
#        print "v_average:"
#        print np.average(v)
#     
#    y_100 = np.loadtxt("u_vs_y_Re=100.txt",usecols=(0,))    
#    u_ghia_100 = np.loadtxt("u_vs_y_Re=100.txt",usecols=(1,))
#    
#    x_100 = np.loadtxt("v_vs_x_Re=100.txt",usecols=(0,))    
#    v_ghia_100 = np.loadtxt("v_vs_x_Re=100.txt",usecols=(1,))
#    
#    n = 100
#    xu = 0.5*np.ones(n)
#    yu = np.linspace(xy_no[:,1].min(),xy_no[:,1].max(),n)
#    #XU,YU = np.meshgrid(xu,yu)
#    # interpolate Z values on defined grid
#    U = griddata(np.vstack((xy_no[:,0].flatten(),xy_no[:,1].flatten())).T, np.vstack(u.flatten()),(xu,yu),method="cubic")
#    
#    yv = 0.5*np.ones(n)
#    xv = np.linspace(xy_no[:,0].min(),xy_no[:,0].max(),n)
#    XV,YV = np.meshgrid(xv,yv)
#    
#    V = griddata(np.vstack((xy_no[:,0].flatten(),xy_no[:,1].flatten())).T, np.vstack(v.flatten()),(xv,yv),method="cubic")
#    
#    figure_name = "M1RK2:Variation of u with y at geometric centre.pdf"
#    figure_file = figure_folder + figure_name       
#    fig = plt.figure(figsize=(figwidth,figheight))
#    plt.grid('on',which='both')
#    plt.xlabel("y",fontsize=textFontSize)
#    plt.ylabel(r"u",fontsize=textFontSize,rotation=90)
#    plt.title("u vs y at x=0.5")
#    plt.plot(yu,U,'k-',label="Re=100: Simulation")
#    plt.plot(y_100,u_ghia_100,'r*',label="Re=100:Ghia et al.")
#    plt.savefig(figure_file)
#    plt.close()
#    
#    figure_name = "M1RK2:Variation of v with x at geometric centre.pdf"
#    figure_file = figure_folder + figure_name       
#    fig = plt.figure(figsize=(figwidth,figheight))
#    plt.grid('on',which='both')
#    plt.xlabel("x",fontsize=textFontSize)
#    plt.ylabel(r"v",fontsize=textFontSize,rotation=90)
#    plt.title("v vs x at x=0.5")
#    plt.plot(xv,V,'k-',label="Re=100: Simulation")
#    plt.plot(x_100,v_ghia_100,'r*',label="Re=100:Ghia et al.")
#    plt.savefig(figure_file)
#    plt.close()
#    
#    #plot_data.plot_data(xy_no[:,0],xy_no[:,1],v,"temp")
#    #plot_data.plot_data(xy_no[:,0],xy_no[:,1],u,"temp1")
    
if p2:
    
    print "Part b"
    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)
    
    #####################################################
    ########## Plot Grid Labels / Connectivity ##########
    #####################################################
    #
    #fig_width = 30
    #fig_height = 17
    #textFontSize   = 15
    #gcafontSize    = 32
    #lineWidth      = 2
    #
    #Plot_Node_Labels = True
    #Plot_Face_Labels = True
    #Plot_CV_Labels   = True
    #
    ## the following enables LaTeX typesetting, which will cause the plotting to take forever..
    ## from matplotlib import rc as matplotlibrc
    ## matplotlibrc('text.latex', preamble='\usepackage{color}')
    ## matplotlibrc('text',usetex=True)
    ## matplotlibrc('font', family='serif')
    #
    #mgplx = 0.05*np.abs(max(xy_no[:,0])-min(xy_no[:,0]))
    #mgply = 0.05*np.abs(max(xy_no[:,1])-min(xy_no[:,1]))
    #xlimits = [min(xy_no[:,0])-mgplx,max(xy_no[:,0])+mgplx]
    #ylimits = [min(xy_no[:,1])-mgply,max(xy_no[:,1])+mgply]
    #
    #fig = plt.figure(0,figsize=(fig_width,fig_height))
    #ax = fig.add_subplot(111)
    #ax.plot(xy_no[:,0],xy_no[:,1],'o',markersize=5,markerfacecolor='k')
    #
    #node_color = 'k'
    #centroid_color = 'r'
    #
    #for inos_of_fa in noofa:
    #   ax.plot(xy_no[inos_of_fa,0], xy_no[inos_of_fa,1], 'k-', linewidth = lineWidth)
    #
    #if Plot_Face_Labels:
    #  nfa = xy_fa.shape[0] # number of faces
    #  faces_indexes = range(0,nfa)
    #  for x_fa,y_fa,ifa in zip(xy_fa[:,0],xy_fa[:,1],faces_indexes):
    #    ax.text(x_fa,y_fa,repr(ifa),transform=ax.transData,color='k',
    #        verticalalignment='center',horizontalalignment='center',fontsize=textFontSize )
    #
    #if Plot_Node_Labels:
    #  nno = xy_no.shape[0] # number of nodes
    #  node_indexes = range(0,nno)
    #  for xn,yn,ino in zip(xy_no[:,0],xy_no[:,1],node_indexes):
    #    ax.text(xn,yn,repr(ino),transform=ax.transData,color='r',
    #        verticalalignment='top',horizontalalignment='left',fontsize=textFontSize )
    #
    #if Plot_CV_Labels:
    #  ncv = xy_cv.shape[0]  # number of control volumes
    #  cv_indexes = range(0,ncv)
    #  for xcv,ycv,icv in zip(xy_cv[:,0],xy_cv[:,1],cv_indexes):
    #    ax.text(xcv,ycv,repr(icv),transform=ax.transData,color='b',
    #        verticalalignment='top',horizontalalignment='left',fontsize=textFontSize )
    #
    #ax.axis('equal')
    #ax.set_xlim(xlimits)
    #ax.set_ylim(ylimits)
    #ax.set_xlabel(r'$x$',fontsize=1.5*gcafontSize)
    #ax.set_ylabel(r'$y$',fontsize=1.5*gcafontSize)
    #plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    #plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    #fig_name = filename.split('.')[0]+'.pdf'
    #plt.savefig(fig_name)
    #
    #keyboard()
    
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    faono1 = np.array(faono) # Converting faono to an array
    wall = np.where(partofa1=='WALL')   # Vectorized approach to find face belonging to part 'COLD'
    lid = np.where(partofa1=='LID')   # Vectorized approach to find face belonging to part 'HOT'
    solid = np.where(partofa1=='SOLID')   # Vectorized approach to find face belonging to part 'SOLID'
    
    
    nno_int = nno-np.unique(noofa[lid]).size-np.unique(noofa[wall]).size # No. of internal nodes
    nfa_int = np.size(np.where(partofa1=='SOLID'))
    
    u = np.zeros(nno)
    v = np.zeros(nno)
    
    u_temp = np.zeros(nno)
    v_temp = np.zeros(nno)
    
    p = np.zeros(ncv)
    
    u[np.unique(noofa[lid])] = lid_v
    
    delx, dely = Operators.gradient_nn(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa)
    
    Divx, Divy = Operators.Divergence_facv(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa)
    
    Gradx, Grady = Operators.Gradient_cvfa(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa1)
    
    Gradxp, Gradyp = Operators.Gradient_cvno(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa1)
    
    DivGrad = (Divx*Gradx) + (Divy*Grady)
    
    DivGrad[0,:] = 0.
    DivGrad[0,0] = 1.
    
#    Avg_n2fa = Operators.Avg_nfa_int(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa)
       
    laplace = (delx*delx) + (dely*dely)
    laplace[nno_int:,:] = 0.
    
    delx[nno_int:,:] = 0.
    dely[nno_int:,:] = 0.
    
    t = 0.
    it = 1
    
    while t<t_final and it<=10: # Currently hard coded to stop after 10 iterations to obtain graphs before blow-up. The and statement can be
    # removed to see the blow-up
        
        u_temp = u
        v_temp = v
    
        u_fa = np.zeros(nfa)
        v_fa = np.zeros(nfa)
        
        u_star = np.zeros(nno)
        v_star = np.zeros(nno)
        
        u_star_fa = np.zeros(nfa)
        v_star_fa = np.zeros(nfa)
        
        
        R_viscous_u = nu*(laplace*u)
        R_viscous_v = nu*(laplace*v)
        
        R_convective_u = (u*(delx*u)) + (v*(dely*u))
        R_convective_v = (u*(delx*v)) + (v*(dely*v))
        
        u_star = u  + (dt*(-R_convective_u + R_viscous_u))
        v_star = v  + (dt*(-R_convective_v + R_viscous_v))
        
        u_star_fa = Avg_n2fa*u_star
        v_star_fa = Avg_n2fa*v_star
        
        #    for i in np.arange(nfa_int):
        #        nodes = noofa[i]
        #        u_star_fa[i] = (u_star[nodes[0]] + u_star[nodes[1]])/2.
        #        v_star_fa[i] = (v_star[nodes[0]] + v_star[nodes[1]])/2.
        #        
        u_star_fa[lid] = lid_v
        v_star_fa[lid] = 0.
        
        u_star_fa[wall] = 0.
        v_star_fa[wall] = 0.
        
        #    keyboard()
        
        q = ((Divx*u_star_fa) + (Divy*v_star_fa))/dt
        q[0] = 0.
        
        p = splinalg.spsolve(DivGrad,q)
        
        u = u_star - (dt*(Gradxp*p))
        
        v = v_star - (dt*(Gradyp*p))
        
#        if np.isnan(np.average(u))==True | np.isnan(np.average(v))==True:
#            
#            u = u_temp
#            v = v_temp
#            break
        
        print "Time %f" %(t)    
        t += dt
        print "Iteration %d" %(it)
        
        it += 1
        
        print "u_average:"
        print np.average(u)
        
        print "v_average:"
        print np.average(v)
         
    y_100 = np.loadtxt("u_vs_y_Re=100.txt",usecols=(0,))    
    u_ghia_100 = np.loadtxt("u_vs_y_Re=100.txt",usecols=(1,))
    
    x_100 = np.loadtxt("v_vs_x_Re=100.txt",usecols=(0,))    
    v_ghia_100 = np.loadtxt("v_vs_x_Re=100.txt",usecols=(1,))
    
    n = 100
    xu = 0.5*np.ones(n)
    yu = np.linspace(xy_no[:,1].min(),xy_no[:,1].max(),n)
    #XU,YU = np.meshgrid(xu,yu)
    # interpolate Z values on defined grid
    U = griddata(np.vstack((xy_no[:,0].flatten(),xy_no[:,1].flatten())).T, np.vstack(u.flatten()),(xu,yu),method="cubic")
    
    yv = 0.5*np.ones(n)
    xv = np.linspace(xy_no[:,0].min(),xy_no[:,0].max(),n)
    XV,YV = np.meshgrid(xv,yv)
    
    V = griddata(np.vstack((xy_no[:,0].flatten(),xy_no[:,1].flatten())).T, np.vstack(v.flatten()),(xv,yv),method="cubic")
    
    figure_name = "M2:Variation of u with y at geometric centre.pdf"
    figure_file = figure_folder + figure_name       
    fig = plt.figure(figsize=(figwidth,figheight))
    plt.grid('on',which='both')
    plt.xlabel("y",fontsize=textFontSize)
    plt.ylabel(r"u",fontsize=textFontSize,rotation=90)
    plt.title("u vs y at x=0.5")
    plt.plot(yu,U,'k-',label="Re=100: Simulation")
    plt.plot(y_100,u_ghia_100,'r*',label="Re=100:Ghia et al.")
    plt.savefig(figure_file)
    plt.close()
    
    figure_name = "M2:Variation of v with x at geometric centre.pdf"
    figure_file = figure_folder + figure_name       
    fig = plt.figure(figsize=(figwidth,figheight))
    plt.grid('on',which='both')
    plt.xlabel("x",fontsize=textFontSize)
    plt.ylabel(r"v",fontsize=textFontSize,rotation=90)
    plt.title("v vs x at x=0.5")
    plt.plot(xv,V,'k-',label="Re=100: Simulation")
    plt.plot(x_100,v_ghia_100,'r*',label="Re=100:Ghia et al.")
    plt.savefig(figure_file)
    plt.close()
    
    print ".............................................................................."
    
    #plot_data.plot_data(xy_no[:,0],xy_no[:,1],v,"temp")
    #plot_data.plot_data(xy_no[:,0],xy_no[:,1],u,"temp1")
    
if p3:
    
    print "Part c"
    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)
    
    #####################################################
    ########## Plot Grid Labels / Connectivity ##########
    #####################################################
    #
    #fig_width = 30
    #fig_height = 17
    #textFontSize   = 15
    #gcafontSize    = 32
    #lineWidth      = 2
    #
    #Plot_Node_Labels = True
    #Plot_Face_Labels = True
    #Plot_CV_Labels   = True
    #
    ## the following enables LaTeX typesetting, which will cause the plotting to take forever..
    ## from matplotlib import rc as matplotlibrc
    ## matplotlibrc('text.latex', preamble='\usepackage{color}')
    ## matplotlibrc('text',usetex=True)
    ## matplotlibrc('font', family='serif')
    #
    #mgplx = 0.05*np.abs(max(xy_no[:,0])-min(xy_no[:,0]))
    #mgply = 0.05*np.abs(max(xy_no[:,1])-min(xy_no[:,1]))
    #xlimits = [min(xy_no[:,0])-mgplx,max(xy_no[:,0])+mgplx]
    #ylimits = [min(xy_no[:,1])-mgply,max(xy_no[:,1])+mgply]
    #
    #fig = plt.figure(0,figsize=(fig_width,fig_height))
    #ax = fig.add_subplot(111)
    #ax.plot(xy_no[:,0],xy_no[:,1],'o',markersize=5,markerfacecolor='k')
    #
    #node_color = 'k'
    #centroid_color = 'r'
    #
    #for inos_of_fa in noofa:
    #   ax.plot(xy_no[inos_of_fa,0], xy_no[inos_of_fa,1], 'k-', linewidth = lineWidth)
    #
    #if Plot_Face_Labels:
    #  nfa = xy_fa.shape[0] # number of faces
    #  faces_indexes = range(0,nfa)
    #  for x_fa,y_fa,ifa in zip(xy_fa[:,0],xy_fa[:,1],faces_indexes):
    #    ax.text(x_fa,y_fa,repr(ifa),transform=ax.transData,color='k',
    #        verticalalignment='center',horizontalalignment='center',fontsize=textFontSize )
    #
    #if Plot_Node_Labels:
    #  nno = xy_no.shape[0] # number of nodes
    #  node_indexes = range(0,nno)
    #  for xn,yn,ino in zip(xy_no[:,0],xy_no[:,1],node_indexes):
    #    ax.text(xn,yn,repr(ino),transform=ax.transData,color='r',
    #        verticalalignment='top',horizontalalignment='left',fontsize=textFontSize )
    #
    #if Plot_CV_Labels:
    #  ncv = xy_cv.shape[0]  # number of control volumes
    #  cv_indexes = range(0,ncv)
    #  for xcv,ycv,icv in zip(xy_cv[:,0],xy_cv[:,1],cv_indexes):
    #    ax.text(xcv,ycv,repr(icv),transform=ax.transData,color='b',
    #        verticalalignment='top',horizontalalignment='left',fontsize=textFontSize )
    #
    #ax.axis('equal')
    #ax.set_xlim(xlimits)
    #ax.set_ylim(ylimits)
    #ax.set_xlabel(r'$x$',fontsize=1.5*gcafontSize)
    #ax.set_ylabel(r'$y$',fontsize=1.5*gcafontSize)
    #plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    #plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    #fig_name = filename.split('.')[0]+'.pdf'
    #plt.savefig(fig_name)
    #
    #keyboard()
    
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    faono1 = np.array(faono) # Converting faono to an array
    wall = np.where(partofa1=='WALL')   # Vectorized approach to find face belonging to part 'COLD'
    lid = np.where(partofa1=='LID')   # Vectorized approach to find face belonging to part 'HOT'
    solid = np.where(partofa1=='SOLID')   # Vectorized approach to find face belonging to part 'SOLID'
    
    
    nno_int = nno-np.unique(noofa[lid]).size-np.unique(noofa[wall]).size # No. of internal nodes
    nfa_int = np.size(np.where(partofa1=='SOLID'))
    
    u = np.zeros(nno)
    v = np.zeros(nno)
    
    u_temp = np.zeros(nno)
    v_temp = np.zeros(nno)
    
    p = np.zeros(ncv)
    
    u[np.unique(noofa[lid])] = lid_v
    
    delx, dely = Operators.gradient_nn(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa)
    
    Divx, Divy = Operators.Divergence_facv(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa)
    
    Gradx, Grady = Operators.Gradient_cvfa(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa1)
    
    DivGrad = (Divx*Gradx) + (Divy*Grady)
    
    DivGrad[0,:] = 0.
    DivGrad[0,0] = 1.
    
    Avg_n2fa = Operators.Avg_nfa_int(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa)
    
    Avgfa2n = Operators.Avg_fan_int(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa)
    
    Gradx_new = Avgfa2n*Gradx
    Grady_new = Avgfa2n*Grady
       
    laplace = (delx*delx) + (dely*dely)
    laplace[nno_int:,:] = 0.
    
    delx[nno_int:,:] = 0.
    dely[nno_int:,:] = 0.
    
    t = 0.
    it = 1
    
    while t<t_final and it<=120: # Currently hard coded to stop after 120 iterations to obtain graphs before blow-up. The and statement can be
    # removed to see the blow-up
        
        u_temp = u
        v_temp = v
    
        u_fa = np.zeros(nfa)
        v_fa = np.zeros(nfa)
        
        u_star = np.zeros(nno)
        v_star = np.zeros(nno)
        
        u_star_fa = np.zeros(nfa)
        v_star_fa = np.zeros(nfa)
        
        
        R_viscous_u = nu*(laplace*u)
        R_viscous_v = nu*(laplace*v)
        
        R_convective_u = (u*(delx*u)) + (v*(dely*u))
        R_convective_v = (u*(delx*v)) + (v*(dely*v))
        
        u_star = u  + (dt*(-R_convective_u + R_viscous_u))
        v_star = v  + (dt*(-R_convective_v + R_viscous_v))
        
        u_star_fa = Avg_n2fa*u_star
        v_star_fa = Avg_n2fa*v_star
        
        #    for i in np.arange(nfa_int):
        #        nodes = noofa[i]
        #        u_star_fa[i] = (u_star[nodes[0]] + u_star[nodes[1]])/2.
        #        v_star_fa[i] = (v_star[nodes[0]] + v_star[nodes[1]])/2.
        #        
        u_star_fa[lid] = lid_v
        v_star_fa[lid] = 0.
        
        u_star_fa[wall] = 0.
        v_star_fa[wall] = 0.
        
        #    keyboard()
        
        q = ((Divx*u_star_fa) + (Divy*v_star_fa))/dt
        q[0] = 0.
        
        p = splinalg.spsolve(DivGrad,q)
        
        u = u_star - (dt*(Gradx_new*p))
        
        v = v_star - (dt*(Grady_new*p))
        
        res = np.average(np.abs(u-u_temp)/np.abs(u_temp))
        
        if res<=10**(-6):
            
            u = u_temp
            v = v_temp
            break
        
        print "Time %f" %(t)    
        t += dt
        print "Iteration %d" %(it)
        
        it += 1
        
        print "u_average:"
        print np.average(u)
        
        print "v_average:"
        print np.average(v)
     
    y_100 = np.loadtxt("u_vs_y_Re=100.txt",usecols=(0,))    
    u_ghia_100 = np.loadtxt("u_vs_y_Re=100.txt",usecols=(1,))
    
    x_100 = np.loadtxt("v_vs_x_Re=100.txt",usecols=(0,))    
    v_ghia_100 = np.loadtxt("v_vs_x_Re=100.txt",usecols=(1,))
    
    n = 100
    xu = 0.5*np.ones(n)
    yu = np.linspace(xy_no[:,1].min(),xy_no[:,1].max(),n)
    #XU,YU = np.meshgrid(xu,yu)
    # interpolate Z values on defined grid
    U = griddata(np.vstack((xy_no[:,0].flatten(),xy_no[:,1].flatten())).T, np.vstack(u.flatten()),(xu,yu),method="cubic")
    
    yv = 0.5*np.ones(n)
    xv = np.linspace(xy_no[:,0].min(),xy_no[:,0].max(),n)
    XV,YV = np.meshgrid(xv,yv)
    
    V = griddata(np.vstack((xy_no[:,0].flatten(),xy_no[:,1].flatten())).T, np.vstack(v.flatten()),(xv,yv),method="cubic")
    
    figure_name = "M3:Variation of u with y at geometric centre.pdf"
    figure_file = figure_folder + figure_name       
    fig = plt.figure(figsize=(figwidth,figheight))
    plt.grid('on',which='both')
    plt.xlabel("y",fontsize=textFontSize)
    plt.ylabel(r"u",fontsize=textFontSize,rotation=90)
    plt.title("u vs y at x=0.5")
    plt.plot(yu,U,'k-',label="Re=100: Simulation")
    plt.plot(y_100,u_ghia_100,'r*',label="Re=100:Ghia et al.")
    plt.savefig(figure_file)
    plt.close()
    
    figure_name = "M3:Variation of v with x at geometric centre.pdf"
    figure_file = figure_folder + figure_name       
    fig = plt.figure(figsize=(figwidth,figheight))
    plt.grid('on',which='both')
    plt.xlabel("x",fontsize=textFontSize)
    plt.ylabel(r"v",fontsize=textFontSize,rotation=90)
    plt.title("v vs x at x=0.5")
    plt.plot(xv,V,'k-',label="Re=100: Simulation")
    plt.plot(x_100,v_ghia_100,'r*',label="Re=100:Ghia et al.")
    plt.savefig(figure_file)
    plt.close()
    
    #plot_data.plot_data(xy_no[:,0],xy_no[:,1],v,"temp")
    #plot_data.plot_data(xy_no[:,0],xy_no[:,1],u,"temp1")
    


