# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:37:46 2017

@author: tanmay
"""
#%%
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
import plot_data
from scipy.interpolate import griddata
from timeit import default_timer
import Operators
from scipy import signal




plt.close('all')

figure_folder = "../report/figures/"
icemcfd_project_folder = './mesh/'
filename = 'circle3mfiner.msh'
mshfile_fullpath = icemcfd_project_folder + filename

#fs = True # Flux splitting algorithm


figwidth,figheight = 14,12
lineWidth = 3
fontSize = 25
gcafontSize = 21
textFontSize = 14

part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)

#keyboard()

####################################################
######### Plot Grid Labels / Connectivity ##########
####################################################

fig_width = 30
fig_height = 17
textFontSize   = 15
gcafontSize    = 32
lineWidth      = 2

t_final = 6.
dt = 0.01
a = 1.
rho = 1.

r_gauss = 0.1
r_sponge = 2.7
factor = 1./(3.-r_sponge)**2
freq = 1.

t1 = 1.
t2 = 1.5
t3 = 2.
tc = 0.


# Works for these on circle3m.msh with RK2

#t_final = 3.
#dt = 0.01
#a = 1.
#rho = 1.
#
#r_gauss = 0.5
#r_sponge = 2.8
#factor = 1./(3.-r_sponge)**2
#freq = 1.

mesh_plot = True

if mesh_plot:
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
#%%
#    keyboard()

nno = xy_no.shape[0]  # No. of nodes
ncv = xy_cv.shape[0]  # No. of CVs
nfa = xy_fa.shape[0]  # No. of faces
partofa1 = np.array(partofa) # Converting partofa to an array
faono1 = np.array(faono) # Converting faono to an array
hot = np.where(partofa1=='HOT')   # Vectorized approach to find face belonging to part 'HOT'
solid = np.where(partofa1=='SOLID')   # Vectorized approach to find face belonging to part 'SOLID'

r_fa = np.sqrt((xy_fa[:,0]**2) + (xy_fa[:,1]**2))
r_cv = np.sqrt((xy_cv[:,0]**2) + (xy_cv[:,1]**2))

k_vel = np.zeros(nfa)
k_p = np.zeros(ncv)

k_vel[np.where(r_fa>r_sponge)] = ((r_fa-r_sponge)**2)/factor
k_p[np.where(r_cv>r_sponge)] = ((r_cv-r_sponge)**2)/factor

nno_int = nno-np.unique(noofa[hot]).size # No. of internal nodes
nfa_int = np.size(np.where(partofa1=='SOLID'))

RK1 = False
RK2 = False
RK4 = True
plotting = True
freqplot = 100

#u = np.zeros(nno)
#v = np.zeros(nno)

u_fa = np.zeros(nfa)
v_fa = np.zeros(nfa)

p = np.zeros(ncv)

l1 = np.where(np.abs(r_cv-t1)==np.min(np.abs(r_cv-t1)))
l2 = np.where(np.abs(r_cv-t2)==np.min(np.abs(r_cv-t2)))
l3 = np.where(np.abs(r_cv-t3)==np.min(np.abs(r_cv-t3)))
lc = np.where(np.abs(r_cv-tc)==np.min(np.abs(r_cv-tc)))

ps1 = np.zeros(int(t_final/dt) + 1)
ps2 = np.zeros(int(t_final/dt) + 1)
ps3 = np.zeros(int(t_final/dt) + 1)
psc = np.zeros(int(t_final/dt) + 1)

#p=np.exp(-((xy_cv[:,0])**2+(xy_cv[:,1])**2)/2.0)

#p[np.where(r_cv<0.2)] = np.exp(-1/(1-(r_cv[np.where(r_cv<0.2)]/0.2)**2))

#p[np.where(r_cv<0.1)] = 1.

#u[np.unique(noofa[hot])] = 0.
#v[np.unique(noofa[hot])] = 0.

Divx, Divy = Operators.Divergence_facv(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa)

Gradx, Grady = Operators.Gradient_cvfa(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa1)

#Avg_n2fa = Operators.Avg_nfa_int(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa)

Avgfa2n = Operators.Avg_fan_int(part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa)


t = 0.
it = 0

#plot_data.plot_data(xy_cv[:,0],xy_cv[:,1],p,"p=%d"%it+".png")

#keyboard()

plot_data.plot_data(xy_cv[:,0],xy_cv[:,1],p,"p=%d"%it+".png")

#%%

while t_final>t:
    
#    u_fa = Avg_n2fa*u
#    v_fa = Avg_n2fa*v
    p[np.where(r_cv<r_gauss)] = (10**(-4))*np.exp(-1/(1-(r_cv[np.where(r_cv<r_gauss)]/r_gauss)**2))*np.sin(2.*np.pi*freq*t)
    
    if RK1:
        
        u_fa -= (dt/rho)*(Gradx*p) + (dt*k_vel*u_fa)
        v_fa -= (dt/rho)*(Grady*p) + (dt*k_vel*v_fa)
    
        p -= (dt*rho*(a**2)*((Divx*u_fa) + (Divy*v_fa))) + (dt*k_p*p)
        
    if RK2:    
        u_fa1 = u_fa - (dt/rho)*(Gradx*p) - (dt*k_vel*u_fa)
        v_fa1 = v_fa - (dt/rho)*(Grady*p) - (dt*k_vel*v_fa)
    
        p1 = p - (dt*rho*(a**2)*((Divx*u_fa) + (Divy*v_fa))) - (dt*k_p*p)
        
        u_fa -= (dt/(2.*rho))*(Gradx*(p + p1)) + ((dt*(k_vel/2.))*(u_fa + u_fa1))
        v_fa -= (dt/(2.*rho))*(Grady*(p + p1)) + ((dt*(k_vel/2.))*(v_fa + v_fa1))
    
        p -= ((dt/2.)*rho*(a**2)*((Divx*(u_fa + u_fa1)) + (Divy*(v_fa + v_fa1)))) + ((dt*(k_p/2.))*(p + p1))
    
    if RK4: 
        
        p1 = p - (((dt/2.)*rho*(a**2)*((Divx*u_fa) + (Divy*v_fa))) + (dt*k_p*p/2.))

        u_fa1 = u_fa - ((dt/(2.*rho))*(Gradx*p) + (dt*k_vel*u_fa))
        v_fa1 = v_fa - ((dt/(2.*rho))*(Grady*p) + (dt*k_vel*v_fa))

        p2 = p - (((dt/2.)*rho*(a**2)*((Divx*u_fa1) + (Divy*v_fa1))) + (dt*k_p*p1/2.))
        
        u_fa2 = u_fa - ((dt/(2.*rho))*(Gradx*p1) + (dt*k_vel*u_fa1))
        v_fa2 = v_fa - ((dt/(2.*rho))*(Grady*p1) + (dt*k_vel*v_fa1))

        p3 = p - ((dt*rho*(a**2)*((Divx*u_fa2) + (Divy*v_fa2))) + (dt*k_p*p2))
        
        u_fa3 = u_fa - ((dt/rho)*(Gradx*p2) + (dt*k_vel*u_fa2))
        v_fa3 = v_fa - ((dt/rho)*(Grady*p2) + (dt*k_vel*v_fa2))

        p -= ((dt/6.)*rho*(a**2)*((Divx*(u_fa + (2.*u_fa1) + (2.*u_fa2) + u_fa3)) + (Divy*(v_fa + (2.*v_fa1) + (2.*v_fa2) + v_fa3)))) + ((dt*k_p/6.)*(p + (2.*p1) + (2.*p2) + p3))
        
        u_fa -= (dt/(6.*rho))*(Gradx*(p + (2.*p1) + (2.*p2) + p3)) + ((dt*k_vel/6.)*(u_fa + (2.*u_fa1) + (2.*u_fa2) + u_fa3))
        v_fa -= (dt/(6.*rho))*(Grady*(p + (2.*p1) + (2.*p2) + p3)) + ((dt*k_vel/6.)*(v_fa + (2.*v_fa1) + (2.*v_fa2) + v_fa3))

       # keyboard()
    
    
#    u = Avgfa2n*u_fa
#    v = Avgfa2n*v_fa 
    
    
    ps1[it] = p[l1]
    ps2[it] = p[l2]
    ps3[it] = p[l3]
    psc[it] = p[lc]

    print "Time %f" %(t)    
    t += dt
    print "Iteration %d" %(it)
    
    print "u_average:"
    print np.average(u_fa)
    
    print "v_average:"
    print np.average(v_fa)
    
    it += 1
    
    
    if plotting:
        if it%freqplot==0:
            plot_data.plot_data(xy_cv[:,0],xy_cv[:,1],p,"p=%d"%it+".png")
            
            fig = plt.figure(1000)
            plt.quiver(xy_fa[:,0],xy_fa[:,1],u_fa,v_fa)
            figure_name = figure_folder + "V=%d"%it+".png"
            plt.savefig(figure_name)
            plt.close()    


print ".............................................................................."
 
#%%

T = np.linspace(0,t_final,it)
fig = plt.figure(1001)
plt.plot(T,ps1,'k-',label='r=1')
plt.plot(T,ps2,'r-',label='r=1.5')
plt.plot(T,ps3,'b-',label='r=2')
plt.xlabel('Pressure')
plt.ylabel('Time')
plt.plot(T,0.1*psc,'k--',label='r=0')
plt.legend(loc='best')

figure_name = figure_folder + "Pressure variation at r=1, 1.5 and 2.pdf"
plt.savefig(figure_name)
plt.close()

#%%
f1, Pxx1 = signal.periodogram(ps1,100)
f2, Pxx2 = signal.periodogram(ps2,1./dt)
f3, Pxx3 = signal.periodogram(ps3,1./dt)
fc, Pxxc = signal.periodogram(psc,1./dt)

fig = plt.figure(1002)
plt.semilogy(f1,Pxx1,'k-',label='r=1')
plt.semilogy(f2,Pxx2,'r-',label='r=1.5')
plt.semilogy(f3,Pxx3,'b-',label='r=2')
plt.semilogy(fc,Pxxc,'k--',label='r=0')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.ylim(10**-18,10**-9) 
plt.xlim(0,10)
plt.legend(loc='best')

figure_name = figure_folder + "Power density spectrum.pdf"
plt.savefig(figure_name)
plt.close()
    

    
    
    
    

    
    

