import os
import sys
import numpy as np   # library to handle arrays like Matlab
import scipy.sparse as scysparse
from pdb import set_trace as keyboard # pdb package allows you to interrupt the python script with the keyboard() command
import spatial_discretization as sd # like include .h file in C++, calls another file
from matplotlib import pyplot as plt
from matplotlib import rc as matplotlibrc

figure_folder = "../report/"

#Latex setup
matplotlibrc('text.latex', preamble='\usepackage{color}')
matplotlibrc('text',usetex=True)
matplotlibrc('font', family='serif')

Plot_Problem1 = True
Plot_Problem2 = True
Plot_Problem3 = True
# ...

############################################################
############################################################

if Plot_Problem1:

    #Definitions for ranges of l, r dx and initializations
    dx_range = np.logspace(-5,0,num=1000, base=10)
    discretization_error = np.zeros((dx_range.size,12)) 
    dx_inverse = 1/dx_range
    x_eval  = 0.0   #Evaluation point
    dfdx_analytical = 5.*np.cos(5.*x_eval + 1.5)*np.tanh(x_eval) + (np.sin(5.*x_eval + 1.5)/np.cosh(x_eval)) #Function evaluation
    
    l = 1
    r = 1
    for idx,dx in enumerate(dx_range):
    
        #Define stencil for a given l and r
        x_stencil_C = np.linspace(-l*dx,r*dx,l+r+1) #collocated
        x_stencil_S = np.linspace((-l+0.5)*dx,(r-0.5)*dx,l+r) #staggered
        
    #   Function evaluation over the stencil
        f_C = np.tanh(x_stencil_C)*np.sin(5.*x_stencil_C + 1.5)
        f_S = np.tanh(x_stencil_S)*np.sin(5.*x_stencil_S + 1.5)
        
        
        order_derivative = 1
        w_der0_C = sd.Generate_Weights(x_stencil_C,x_eval,order_derivative)
        w_der0_S = sd.Generate_Weights(x_stencil_S,x_eval,order_derivative)
       
        dfdx_hat_C = w_der0_C.dot(f_C)
        dfdx_hat_S = w_der0_S.dot(f_S)
    #   Can print output for every iteration in the loop
    #                print "dfdx_hat_C:        " + "%2.18f" %dfdx_hat_C
    #                print "dfdx_hat_C:        " + "%2.18f" %dfdx_hat_S
    #                print "dfdx_analytical: " + "%2.18f" %dfdx_analytical
    
    #   Absolute truncation error for both schemes
        discretization_error[idx,0] = np.abs(dfdx_hat_C-dfdx_analytical)
        discretization_error[idx,6] = np.abs(dfdx_hat_S-dfdx_analytical)
        
    l = 2
    r = 2
    for idx,dx in enumerate(dx_range):
    
        #Define stencil for a given l and r
        x_stencil_C = np.linspace(-l*dx,r*dx,l+r+1) #collocated
        x_stencil_S = np.linspace((-l+0.5)*dx,(r-0.5)*dx,l+r) #staggered
        
    #   Function evaluation over the stencil
        f_C = np.tanh(x_stencil_C)*np.sin(5.*x_stencil_C + 1.5)
        f_S = np.tanh(x_stencil_S)*np.sin(5.*x_stencil_S + 1.5)
        
        
        order_derivative = 1
        w_der0_C = sd.Generate_Weights(x_stencil_C,x_eval,order_derivative)
        w_der0_S = sd.Generate_Weights(x_stencil_S,x_eval,order_derivative)
       
        dfdx_hat_C = w_der0_C.dot(f_C)
        dfdx_hat_S = w_der0_S.dot(f_S)
    #   Can print output for every iteration in the loop
    #                print "dfdx_hat_C:        " + "%2.18f" %dfdx_hat_C
    #                print "dfdx_hat_C:        " + "%2.18f" %dfdx_hat_S
    #                print "dfdx_analytical: " + "%2.18f" %dfdx_analytical
    
    #   Absolute truncation error for both schemes
        discretization_error[idx,1] = np.abs(dfdx_hat_C-dfdx_analytical)
        discretization_error[idx,7] = np.abs(dfdx_hat_S-dfdx_analytical)
        
    l = 3
    r = 3
    for idx,dx in enumerate(dx_range):
    
        #Define stencil for a given l and r
        x_stencil_C = np.linspace(-l*dx,r*dx,l+r+1) #collocated
        x_stencil_S = np.linspace((-l+0.5)*dx,(r-0.5)*dx,l+r) #staggered
        
    #   Function evaluation over the stencil
        f_C = np.tanh(x_stencil_C)*np.sin(5.*x_stencil_C + 1.5)
        f_S = np.tanh(x_stencil_S)*np.sin(5.*x_stencil_S + 1.5)
        
        
        order_derivative = 1
        w_der0_C = sd.Generate_Weights(x_stencil_C,x_eval,order_derivative)
        w_der0_S = sd.Generate_Weights(x_stencil_S,x_eval,order_derivative)
       
        dfdx_hat_C = w_der0_C.dot(f_C)
        dfdx_hat_S = w_der0_S.dot(f_S)
    #   Can print output for every iteration in the loop
    #                print "dfdx_hat_C:        " + "%2.18f" %dfdx_hat_C
    #                print "dfdx_hat_C:        " + "%2.18f" %dfdx_hat_S
    #                print "dfdx_analytical: " + "%2.18f" %dfdx_analytical
    
    #   Absolute truncation error for both schemes
        discretization_error[idx,2] = np.abs(dfdx_hat_C-dfdx_analytical)
        discretization_error[idx,8] = np.abs(dfdx_hat_S-dfdx_analytical)
        
    l = 0
    r = 1
    l_s  = l + 1
    for idx,dx in enumerate(dx_range):
    
        #Define stencil for a given l and r
        x_stencil_C = np.linspace(-l*dx,r*dx,l+r+1) #collocated
        x_stencil_S = np.linspace((-l_s+0.5)*dx,(r-0.5)*dx,l_s+r) #staggered
        
    #   Function evaluation over the stencil
        f_C = np.tanh(x_stencil_C)*np.sin(5.*x_stencil_C + 1.5)
        f_S = np.tanh(x_stencil_S)*np.sin(5.*x_stencil_S + 1.5)
        
        
        order_derivative = 1
        w_der0_C = sd.Generate_Weights(x_stencil_C,x_eval,order_derivative)
        w_der0_S = sd.Generate_Weights(x_stencil_S,x_eval,order_derivative)
       
        dfdx_hat_C = w_der0_C.dot(f_C)
        dfdx_hat_S = w_der0_S.dot(f_S)
    #   Can print output for every iteration in the loop
    #                print "dfdx_hat_C:        " + "%2.18f" %dfdx_hat_C
    #                print "dfdx_hat_C:        " + "%2.18f" %dfdx_hat_S
    #                print "dfdx_analytical: " + "%2.18f" %dfdx_analytical
    
    #   Absolute truncation error for both schemes
        discretization_error[idx,3] = np.abs(dfdx_hat_C-dfdx_analytical)
        discretization_error[idx,9] = np.abs(dfdx_hat_S-dfdx_analytical)
        
    l = 0
    r = 2
    l_s = l + 1
    for idx,dx in enumerate(dx_range):
    
        #Define stencil for a given l and r
        x_stencil_C = np.linspace(-l*dx,r*dx,l+r+1) #collocated
        x_stencil_S = np.linspace((-l_s+0.5)*dx,(r-0.5)*dx,l_s+r) #staggered
        
    #   Function evaluation over the stencil
        f_C = np.tanh(x_stencil_C)*np.sin(5.*x_stencil_C + 1.5)
        f_S = np.tanh(x_stencil_S)*np.sin(5.*x_stencil_S + 1.5)
        
        
        order_derivative = 1
        w_der0_C = sd.Generate_Weights(x_stencil_C,x_eval,order_derivative)
        w_der0_S = sd.Generate_Weights(x_stencil_S,x_eval,order_derivative)
       
        dfdx_hat_C = w_der0_C.dot(f_C)
        dfdx_hat_S = w_der0_S.dot(f_S)
    #   Can print output for every iteration in the loop
    #                print "dfdx_hat_C:        " + "%2.18f" %dfdx_hat_C
    #                print "dfdx_hat_C:        " + "%2.18f" %dfdx_hat_S
    #                print "dfdx_analytical: " + "%2.18f" %dfdx_analytical
    
    #   Absolute truncation error for both schemes
        discretization_error[idx,4] = np.abs(dfdx_hat_C-dfdx_analytical)
        discretization_error[idx,10] = np.abs(dfdx_hat_S-dfdx_analytical)
        
    l = 0
    r = 3
    l_s  = l + 1
    for idx,dx in enumerate(dx_range):
    
        #Define stencil for a given l and r
        x_stencil_C = np.linspace(-l*dx,r*dx,l+r+1) #collocated
        x_stencil_S = np.linspace((-l_s+0.5)*dx,(r-0.5)*dx,l_s+r) #staggered
        
    #   Function evaluation over the stencil
        f_C = np.tanh(x_stencil_C)*np.sin(5.*x_stencil_C + 1.5)
        f_S = np.tanh(x_stencil_S)*np.sin(5.*x_stencil_S + 1.5)
        
        
        order_derivative = 1
        w_der0_C = sd.Generate_Weights(x_stencil_C,x_eval,order_derivative)
        w_der0_S = sd.Generate_Weights(x_stencil_S,x_eval,order_derivative)
       
        dfdx_hat_C = w_der0_C.dot(f_C)
        dfdx_hat_S = w_der0_S.dot(f_S)
    #   Can print output for every iteration in the loop
    #                print "dfdx_hat_C:        " + "%2.18f" %dfdx_hat_C
    #                print "dfdx_hat_C:        " + "%2.18f" %dfdx_hat_S
    #                print "dfdx_analytical: " + "%2.18f" %dfdx_analytical
    
    #   Absolute truncation error for both schemes
        discretization_error[idx,5] = np.abs(dfdx_hat_C-dfdx_analytical)
        discretization_error[idx,11] = np.abs(dfdx_hat_S-dfdx_analytical)
    
        
#Plotting code
        
    figure_name_1 = "Absolute truncation error vs inverse grid spacing for Collocated Schemes problem 1.pdf"
    figure_name_2 = "Absolute truncation error vs inverse grid spacing for Staggered Schemes problem 1.pdf"
    figwidth       = 20
    figheight      = 16
    lineWidth      = 3
    textFontSize   = 14
    gcafontSize    = 14
    
    fig1 = plt.figure(0, figsize=(figwidth,figheight))
    ax_1 = fig1.add_subplot(2,3,1)
    ax_2 = fig1.add_subplot(2,3,2)
    ax_3 = fig1.add_subplot(2,3,3)
    ax_4 = fig1.add_subplot(2,3,4)
    ax_5 = fig1.add_subplot(2,3,5)
    ax_6 = fig1.add_subplot(2,3,6)
    
    
    
    
#Collocated scheme plots
    ax = ax_1
    plt.axes(ax)
    ax.loglog(dx_inverse,discretization_error[:,0],'-k',linewidth=lineWidth)
    ax.loglog(dx_inverse,dx_range,'m--')
    ax.loglog(dx_inverse,dx_range**2,'r--')
    ax.loglog(dx_inverse,dx_range**3,'g--')
    ax.loglog(dx_inverse,dx_range**4,'b--')
    ax.loglog(dx_inverse,dx_range**5,'k--')
    ax.loglog(dx_inverse,dx_range**6,'c--')
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.grid('on',which='both')
  #  ax.set_xticks()
  #  ax.set_xlim()
  #  ax.set_yticks()
  #  ax.set_ylim()
    ax.set_xlabel(r"$dx^{-1}$",fontsize=textFontSize)
    ax.set_ylabel(r"$epsilon_{TR}$",fontsize=textFontSize,rotation=90)
    plt.title("l=1, r=1")
    plt.legend(["l=1,r=1","Order 1","Order 2","Order 3","Order 4","Order 5","Order 6"])
    
    ax = ax_2
    plt.axes(ax)
    ax.loglog(dx_inverse,discretization_error[:,1],'-k',linewidth=lineWidth)
    ax.loglog(dx_inverse,dx_range,'m--')
    ax.loglog(dx_inverse,dx_range**2,'r--')
    ax.loglog(dx_inverse,dx_range**3,'g--')
    ax.loglog(dx_inverse,dx_range**4,'b--')
    ax.loglog(dx_inverse,dx_range**5,'k--')
    ax.loglog(dx_inverse,dx_range**6,'c--')
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.grid('on',which='both')
  #  ax.set_xticks()
  #  ax.set_xlim()
  #  ax.set_yticks()
  #  ax.set_ylim()
    ax.set_xlabel(r"$dx^{-1}$",fontsize=textFontSize)
    ax.set_ylabel(r"$epsilon_{TR}$",fontsize=textFontSize,rotation=90)
    plt.title("l=2, r=2")
    plt.legend(["l=2,r=2","Order 1","Order 2","Order 3","Order 4","Order 5","Order 6"])
    
    ax = ax_3
    plt.axes(ax)
    ax.loglog(dx_inverse,discretization_error[:,2],'-k',linewidth=lineWidth)
    ax.loglog(dx_inverse,dx_range,'m--')
    ax.loglog(dx_inverse,dx_range**2,'r--')
    ax.loglog(dx_inverse,dx_range**3,'g--')
    ax.loglog(dx_inverse,dx_range**4,'b--')
    ax.loglog(dx_inverse,dx_range**5,'k--')
    ax.loglog(dx_inverse,dx_range**6,'c--')
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.grid('on',which='both')
  #  ax.set_xticks()
  #  ax.set_xlim()
  #  ax.set_yticks()
  #  ax.set_ylim()
    ax.set_xlabel(r"$dx^{-1}$",fontsize=textFontSize)
    ax.set_ylabel(r"$epsilon_{TR}$",fontsize=textFontSize,rotation=90)
    plt.title("l=3, r=3")
    plt.legend(["l=3,r=3","Order 1","Order 2","Order 3","Order 4","Order 5","Order 6"])
    
    ax = ax_4
    plt.axes(ax)
    ax.loglog(dx_inverse,discretization_error[:,3],'-k',linewidth=lineWidth)
    ax.loglog(dx_inverse,dx_range,'m--')
    ax.loglog(dx_inverse,dx_range**2,'r--')
    ax.loglog(dx_inverse,dx_range**3,'g--')
    ax.loglog(dx_inverse,dx_range**4,'b--')
    ax.loglog(dx_inverse,dx_range**5,'k--')
    ax.loglog(dx_inverse,dx_range**6,'c--')
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.grid('on',which='both')
  #  ax.set_xticks()
  #  ax.set_xlim()
  #  ax.set_yticks()
  #  ax.set_ylim()
    ax.set_xlabel(r"$dx^{-1}$",fontsize=textFontSize)
    ax.set_ylabel(r"$epsilon_{TR}$",fontsize=textFontSize,rotation=90)
    plt.title("l=0, r=1")
    plt.legend(["l=0,r=1","Order 1","Order 2","Order 3","Order 4","Order 5","Order 6"])
    
    
    ax = ax_5
    plt.axes(ax)
    ax.loglog(dx_inverse,discretization_error[:,4],'-k',linewidth=lineWidth)
    ax.loglog(dx_inverse,dx_range,'m--')
    ax.loglog(dx_inverse,dx_range**2,'r--')
    ax.loglog(dx_inverse,dx_range**3,'g--')
    ax.loglog(dx_inverse,dx_range**4,'b--')
    ax.loglog(dx_inverse,dx_range**5,'k--')
    ax.loglog(dx_inverse,dx_range**6,'c--')
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.grid('on',which='both')
  #  ax.set_xticks()
  #  ax.set_xlim()
  #  ax.set_yticks()
  #  ax.set_ylim()
    ax.set_xlabel(r"$dx^{-1}$",fontsize=textFontSize)
    ax.set_ylabel(r"$epsilon_{TR}$",fontsize=textFontSize,rotation=90)
    plt.title("l=0, r=2")
    plt.legend(["l=0,r=2","Order 1","Order 2","Order 3","Order 4","Order 5","Order 6"])
    
    
    ax = ax_6
    plt.axes(ax)
    ax.loglog(dx_inverse,discretization_error[:,5],'-k',linewidth=lineWidth)
    ax.loglog(dx_inverse,dx_range,'m--')
    ax.loglog(dx_inverse,dx_range**2,'r--')
    ax.loglog(dx_inverse,dx_range**3,'g--')
    ax.loglog(dx_inverse,dx_range**4,'b--')
    ax.loglog(dx_inverse,dx_range**5,'k--')
    ax.loglog(dx_inverse,dx_range**6,'c--')
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.grid('on',which='both')
  #  ax.set_xticks()
  #  ax.set_xlim()
  #  ax.set_yticks()
  #  ax.set_ylim()
    ax.set_xlabel(r"$dx^{-1}$",fontsize=textFontSize)
    ax.set_ylabel(r"$epsilon_{TR}$",fontsize=textFontSize,rotation=90)
    plt.title("l=0, r=3")
    plt.legend(["l=0,r=3","Order 1","Order 2","Order 3","Order 4","Order 5","Order 6"])
    
    figure_file_path_1 = figure_folder + figure_name_1
    print "Saving figure: " + figure_file_path_1
    plt.tight_layout()
    plt.savefig(figure_file_path_1)
    plt.close()


#Staggered scheme 
    fig2 = plt.figure(1, figsize=(figwidth,figheight))    
    ax_7 = fig2.add_subplot(2,3,1)
    ax_8 = fig2.add_subplot(2,3,2)
    ax_9 = fig2.add_subplot(2,3,3)
    ax_10 = fig2.add_subplot(2,3,4)
    ax_11 = fig2.add_subplot(2,3,5)
    ax_12 = fig2.add_subplot(2,3,6)
 
    ax = ax_7
    plt.axes(ax)
    ax.loglog(dx_inverse,discretization_error[:,6],'-k',linewidth=lineWidth)
    ax.loglog(dx_inverse,dx_range,'m--')
    ax.loglog(dx_inverse,dx_range**2,'r--')
    ax.loglog(dx_inverse,dx_range**3,'g--')
    ax.loglog(dx_inverse,dx_range**4,'b--')
    ax.loglog(dx_inverse,dx_range**5,'k--')
    ax.loglog(dx_inverse,dx_range**6,'c--')
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.grid('on',which='both')
  #  ax.set_xticks()
  #  ax.set_xlim()
  #  ax.set_yticks()
  #  ax.set_ylim()
    ax.set_xlabel(r"$dx^{-1}$",fontsize=textFontSize)
    ax.set_ylabel(r"$epsilon_{TR}$",fontsize=textFontSize,rotation=90)
    plt.title("l=1, r=1")
    plt.legend(["l=1,r=1","Order 1","Order 2","Order 3","Order 4","Order 5","Order 6"])
    
    ax = ax_8
    plt.axes(ax)
    ax.loglog(dx_inverse,discretization_error[:,7],'-k',linewidth=lineWidth)
    ax.loglog(dx_inverse,dx_range,'m--')
    ax.loglog(dx_inverse,dx_range**2,'r--')
    ax.loglog(dx_inverse,dx_range**3,'g--')
    ax.loglog(dx_inverse,dx_range**4,'b--')
    ax.loglog(dx_inverse,dx_range**5,'k--')
    ax.loglog(dx_inverse,dx_range**6,'c--')
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.grid('on',which='both')
  #  ax.set_xticks()
  #  ax.set_xlim()
  #  ax.set_yticks()
  #  ax.set_ylim()
    ax.set_xlabel(r"$dx^{-1}$",fontsize=textFontSize)
    ax.set_ylabel(r"$epsilon_{TR}$",fontsize=textFontSize,rotation=90)
    plt.title("l=2, r=2")
    plt.legend(["l=2,r=2","Order 1","Order 2","Order 3","Order 4","Order 5","Order 6"])
    
    ax = ax_9
    plt.axes(ax)
    ax.loglog(dx_inverse,discretization_error[:,8],'-k',linewidth=lineWidth)
    ax.loglog(dx_inverse,dx_range,'m--')
    ax.loglog(dx_inverse,dx_range**2,'r--')
    ax.loglog(dx_inverse,dx_range**3,'g--')
    ax.loglog(dx_inverse,dx_range**4,'b--')
    ax.loglog(dx_inverse,dx_range**5,'k--')
    ax.loglog(dx_inverse,dx_range**6,'c--')
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.grid('on',which='both')
  #  ax.set_xticks()
  #  ax.set_xlim()
  #  ax.set_yticks()
  #  ax.set_ylim()
    ax.set_xlabel(r"$dx^{-1}$",fontsize=textFontSize)
    ax.set_ylabel(r"$epsilon_{TR}$",fontsize=textFontSize,rotation=90)
    plt.title("l=3, r=3")
    plt.legend(["l=3,r=3","Order 1","Order 2","Order 3","Order 4","Order 5","Order 6"])
    
    ax = ax_10
    plt.axes(ax)
    ax.loglog(dx_inverse,discretization_error[:,9],'-k',linewidth=lineWidth)
    ax.loglog(dx_inverse,dx_range,'m--')
    ax.loglog(dx_inverse,dx_range**2,'r--')
    ax.loglog(dx_inverse,dx_range**3,'g--')
    ax.loglog(dx_inverse,dx_range**4,'b--')
    ax.loglog(dx_inverse,dx_range**5,'k--')
    ax.loglog(dx_inverse,dx_range**6,'c--')
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.grid('on',which='both')
  #  ax.set_xticks()
  #  ax.set_xlim()
  #  ax.set_yticks()
  #  ax.set_ylim()
    ax.set_xlabel(r"$dx^{-1}$",fontsize=textFontSize)
    ax.set_ylabel(r"$epsilon_{TR}$",fontsize=textFontSize,rotation=90)
    plt.title("l=1, r=1")
    plt.legend(["l=1,r=1","Order 1","Order 2","Order 3","Order 4","Order 5","Order 6"])
    
    ax = ax_11
    plt.axes(ax)
    ax.loglog(dx_inverse,discretization_error[:,10],'-k',linewidth=lineWidth)
    ax.loglog(dx_inverse,dx_range,'m--')
    ax.loglog(dx_inverse,dx_range**2,'r--')
    ax.loglog(dx_inverse,dx_range**3,'g--')
    ax.loglog(dx_inverse,dx_range**4,'b--')
    ax.loglog(dx_inverse,dx_range**5,'k--')
    ax.loglog(dx_inverse,dx_range**6,'c--')
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.grid('on',which='both')
  #  ax.set_xticks()
  #  ax.set_xlim()
  #  ax.set_yticks()
  #  ax.set_ylim()
    ax.set_xlabel(r"$dx^{-1}$",fontsize=textFontSize)
    ax.set_ylabel(r"$epsilon_{TR}$",fontsize=textFontSize,rotation=90)
    plt.title("l=1, r=2")
    plt.legend(["l=1,r=2","Order 1","Order 2","Order 3","Order 4","Order 5","Order 6"])
    
    ax = ax_12
    plt.axes(ax)
    ax.loglog(dx_inverse,discretization_error[:,11],'-k',linewidth=lineWidth)
    ax.loglog(dx_inverse,dx_range,'m--')
    ax.loglog(dx_inverse,dx_range**2,'r--')
    ax.loglog(dx_inverse,dx_range**3,'g--')
    ax.loglog(dx_inverse,dx_range**4,'b--')
    ax.loglog(dx_inverse,dx_range**5,'k--')
    ax.loglog(dx_inverse,dx_range**6,'c--')
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.grid('on',which='both')
  #  ax.set_xticks()
  #  ax.set_xlim()
  #  ax.set_yticks()
  #  ax.set_ylim()
    ax.set_xlabel(r"$dx^{-1}$",fontsize=textFontSize)
    ax.set_ylabel(r"$epsilon_{TR}$",fontsize=textFontSize,rotation=90)
    plt.title("l=1, r=3")
    plt.legend(["l=1,r=3","Order 1","Order 2","Order 3","Order 4","Order 5","Order 6"])
    
    
    figure_file_path_2 = figure_folder + figure_name_2
    print "Saving figure: " + figure_file_path_2
    plt.savefig(figure_file_path_2)
    plt.close()
    

    # Use sample script in Homework 0 to plot results in publication quality

############################################################
############################################################

if Plot_Problem2:

#               Part a

#   Initializations

    N = 10   #No. of discretized points
    x_mesh = np.linspace(-3,3,N)   #Grid
    
#Generating operators
    D_13 = sd.Generate_Spatial_Operators(x_mesh,"3rd_order",1)
    D_15 = sd.Generate_Spatial_Operators(x_mesh,"5th_order",1)
    D_33 = sd.Generate_Spatial_Operators(x_mesh,"3rd_order",3)
    D_35 = sd.Generate_Spatial_Operators(x_mesh,"5th_order",3)
    
#spy plot generation
    
        
    figure_name = "Spy Plot for Derivative Operators problem 2a.pdf"
    figwidth       = 20
    figheight      = 16
    lineWidth      = 3
    textFontSize   = 14
    gcafontSize    = 14
    
    fig = plt.figure(0, figsize=(figwidth,figheight))
    ax_1 = fig.add_subplot(2,2,1)
    ax_2 = fig.add_subplot(2,2,2)
    ax_3 = fig.add_subplot(2,2,3)
    ax_4 = fig.add_subplot(2,2,4)
    
    ax = ax_1
    plt.axes(ax)
    plt.title("1st Derivative using 3rd Order Lagrange Polynomial")
    ax.spy(D_13)
    ax = ax_2
    plt.axes(ax)
    plt.title("1st Derivative using 5th Order Lagrange Polynomial")
    ax.spy(D_15)
    ax = ax_3
    plt.axes(ax)
    plt.title("3rd Derivative using 3rd Order Lagrange Polynomial")
    ax.spy(D_33)
    ax = ax_4
    plt.axes(ax)
    plt.title("3rd Derivative using 5th Order Lagrange Polynomial")
    ax.spy(D_35)
    figure_file_path = figure_folder + figure_name
    print "Saving figure: " + figure_file_path
    plt.tight_layout()
    plt.savefig(figure_file_path)
    plt.close()
    
    
##               Part b
##               The function used is log(x) and the fitting done is for both 3rd and 5th order lagrange polynomial
#    
##           Note that 3rd order lagrange polynomial operator has been solved for by considering the derivative at the other end of the boundary (x1)    
#    
##   Domain initialization amnd grid spacing definitions

    N_1 = np.linspace(10,200,80)
    x0 = 15
    x1 = 500
    Dx = np.zeros(N_1.size)
    e_rms_35 = np.zeros(Dx.size)
    e_rms_33 = e_rms_35
    for jn,n in enumerate(N_1):   #looping over all grid spacings
        
        #discretization
        x_mesh_2 = np.linspace(x0,x1,n)
        dx = (x1-x0)/(n-1)
        Dx[jn] = dx
        f = np.zeros(x_mesh_2.size)
        u_35 = f
        u_33 = f
        
        #Change this block and the domain (if needed) before the for loop depending on the function being implemented
        f[2:n-1] = 2/(x_mesh_2[2:n-1]**3)
        f[0] = np.log(x_mesh_2[0])
        f[1] = 1/(x_mesh_2[0])
        f[-1] = 1/(x_mesh_2[-1])
        g = np.log(x_mesh_2)
        
        #Operator Generation using 5th order langrange polynomial
        D_1 = sd.Generate_Spatial_Operators(x_mesh_2,"5th_order",3)
        #Operator Generation using 5th order langrange polynomial
        D_2 = sd.Generate_Spatial_Operators(x_mesh_2,"3rd_order",3)
        D_1 = D_1.todense()
        D_2 = D_2.todense()
        
        
   #Inserting boundary conditions. 1st derivative is based on 1st order finite difference using the immediate neighbouring point
   #1st derivative can also be implemented at the boundary using 5th order Lagrange fitting for 1st derivative
        D_1[0,0] = 1
        D_1[0,1:] = 0
        D_1[1,0] = -1/dx
        D_1[1,1] = 1/dx
        D_1[1,2:] = 0
        D_1[-1,-1] = 1/dx
        D_1[-1,-2] = -1/dx
        D_1[-1,:-2] = 0
        
        D_2[0,0] = 1
        D_2[0,1:] = 0
        D_2[1,0] = -1/dx
        D_2[1,1] = 1/dx
        D_2[1,2:] = 0
        D_2[-1,-1] = 1/dx
        D_2[-1,-2] = -1/dx
        D_2[-1,:-2] = 0
        
#        D_2[0,0] = 1
#        D_2[0,1:] = 0
#        D_3 = sd.Generate_Spatial_Operators(x_mesh_2,"5th_order",1).todense()
#        D_2[1,:] = D_3[0,:]
#        D_2[-1,:] = D_3[-1,:]
        
        #Actual calculations
        A = np.linalg.inv(D_1)
        B = np.linalg.inv(D_2)
        u_35 = np.dot(A,f)
        u_33 = np.dot(B,f)
        
        #Finding RMS error
        e_rms_35[jn] = np.linalg.norm(u_35-g)/np.sqrt(n)
        e_rms_33[jn] = np.linalg.norm(u_33-g)/np.sqrt(n)
    Dx_inv = 1/Dx
    
    
    #Plotting Block
    figure_name = "RMS error vs inverse grid spacing problem 2b.pdf"
    figwidth       = 20
    figheight      = 16
    lineWidth      = 3
    textFontSize   = 28
    gcafontSize    = 28
    
    fig = plt.figure(0, figsize=(figwidth,figheight))
    ax_1 = fig.add_subplot(1,2,1)
    ax_2 = fig.add_subplot(1,2,2)
    ax = ax_1
    plt.axes(ax)
    plt.title("Derivative using 5th order Lagrange Polynomial")
    ax.set_xlabel(r"$dx^{-1}$",fontsize=textFontSize)
    ax.set_ylabel(r"$epsilon_{RMS}$",fontsize=textFontSize,rotation=90)
    ax.loglog(Dx_inv,e_rms_35)
    ax.loglog(Dx_inv,Dx,'m--')
    ax.loglog(Dx_inv,Dx**2,'r--')
    ax.loglog(Dx_inv,Dx**3,'g--')
    ax.loglog(Dx_inv,Dx**4,'b--')
    ax.loglog(Dx_inv,Dx**5,'k--')
    ax.loglog(Dx_inv,Dx**6,'c--')
    plt.legend(["RMS Error","Order 1","Order 2","Order 3","Order 4","Order 5","Order 6"])
    ax = ax_2
    plt.axes(ax)
    plt.title("Derivative using 3th order Lagrange Polynomial")
    ax.set_xlabel(r"$dx^{-1}$",fontsize=textFontSize)
    ax.set_ylabel(r"$epsilon_{RMS}$",fontsize=textFontSize,rotation=90)
    ax.loglog(Dx_inv,e_rms_33)
    ax.loglog(Dx_inv,Dx,'m--')
    ax.loglog(Dx_inv,Dx**2,'r--')
    ax.loglog(Dx_inv,Dx**3,'g--')
    ax.loglog(Dx_inv,Dx**4,'b--')
    ax.loglog(Dx_inv,Dx**5,'k--')
    ax.loglog(Dx_inv,Dx**6,'c--')
    plt.legend(["RMS Error","Order 1","Order 2","Order 3","Order 4","Order 5","Order 6"])
    figure_file_path = figure_folder + figure_name
    print "Saving figure: " + figure_file_path
    plt.tight_layout()
    plt.savefig(figure_file_path)
    plt.close()
    
if Plot_Problem3:
    
    N_1 = np.linspace(10,200,80)
    x0 = 15
    x1 = 500
    Dx = np.zeros(N_1.size)
    e_rms = np.zeros(Dx.size)
    for jn,n in enumerate(N_1):
        #The warning for n not being an integer can be handled
        #domain initialization
        x_mesh_2 = np.linspace(x0,x1,n)
        dx = (x1-x0)/(n-1)
        Dx[jn] = dx
        f = np.zeros(x_mesh_2.size)
        u = f
        
        #Change this block and the domain before the for loop depending on the function being implemented
        f[2:n-1] = 2/(x_mesh_2[2:n-1]**3)
        f[0] = np.log(x_mesh_2[0])
        f[1] = 1/(x_mesh_2[0])
        f[-1] = 1/(x_mesh_2[-1])
        g = np.log(x_mesh_2)
        
        #Operator Generation
        D_35_2 = sd.Generate_Spatial_Operators(x_mesh_2,"5th_order",3)
        D_15_2 = sd.Generate_Spatial_Operators(x_mesh_2,"5th_order",1)
        D_15_1 = D_15_2.todense()
        D_35_1 = D_35_2.todense()
        L = np.identity(x_mesh_2.size)
        R = np.zeros((x_mesh_2.size,x_mesh_2.size))
        for j in np.arange(x_mesh_2.size):
            if j == 0:
                R[j,:] = D_35_1[j,:]
            
            elif j == 1:
                R[j,:] = D_35_1[j,:]
                
            elif j == x_mesh_2.size-1:
                R[j,:] = D_35_1[j,:]
                
            elif j == x_mesh_2.size-2:
                R[j,:] = D_35_1[j,:]
                
            else:
                L[j,j-1] = 0.5
                L[j,j+1] = 0.5
                R[j,j-2] = -1/dx**3
                R[j,j+2] = 1/dx**3
                R[j,j-1] = 2/dx**3
                R[j,j+1] = -2/dx**3
        
        D = np.dot(np.linalg.inv(L),R)
        D[1,:] = D_15_1[0,:] #Using lagrange interpolation for derivatives
        D[-1,:] = D_15_1[-1,:]
        D[0,0] = 1
        D[0,1:] = 0
        
        #Commented section includes 1st order FD for derivatives
#        D[1,0] = -1/dx
#        D[1,1] = 1/dx
#        D[1,2:] = 0
#        D[-1,-1] = 1/dx
#        D[-1,-2] = -1/dx
#        D[-1,:-2] = 0
        D = np.linalg.inv(D)
        u = np.dot(D,f)
        e_rms[jn] = np.linalg.norm(u-g)/np.sqrt(n)
    Dx_inv = 1/Dx
    
    
    figure_name = "RMS error vs inverse grid spacing problem 3.pdf"
    figwidth       = 20
    figheight      = 16
    lineWidth      = 3
    textFontSize   = 28
    gcafontSize    = 28
    fig = plt.figure(0, figsize=(figwidth,figheight))
    ax_1 = fig.add_subplot(1,1,1)
    ax = ax_1
    plt.axes(ax)
    plt.title("RMS error vs inverse grid spacing problem 3")
    ax.set_xlabel(r"$dx^{-1}$",fontsize=textFontSize)
    ax.set_ylabel(r"$epsilon_{RMS}$",fontsize=textFontSize,rotation=90)
    ax.loglog(Dx_inv,e_rms)
    ax.loglog(Dx_inv,Dx,'m--')
    ax.loglog(Dx_inv,Dx**2,'r--')
    ax.loglog(Dx_inv,Dx**3,'g--')
    ax.loglog(Dx_inv,Dx**4,'b--')
    ax.loglog(Dx_inv,Dx**5,'k--')
    ax.loglog(Dx_inv,Dx**6,'c--')
    plt.legend(["RMS Error","Order 1","Order 2","Order 3","Order 4","Order 5","Order 6"])
    figure_file_path= figure_folder + figure_name
    print "Saving figure: " + figure_file_path
    plt.savefig(figure_file_path)
    plt.close()
   
             
            
    
    
#keyboard() # <<<<<
#sys.exit("Nothing here yet...")
