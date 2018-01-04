import os
import sys
import numpy as np
import scipy.sparse as scysparse
from pdb import set_trace as keyboard
from time import sleep
import spatial_discretization
import scipy.sparse as scysparse
import scipy.sparse.linalg as spysparselinalg  # sparse linear algebra
import scipy.linalg as scylinalg               # non-sparse linear algebra
from matplotlib import rc as matplotlibrc
import solve
from matplotlib import pyplot as plt

#Latex-related definitions
matplotlibrc('text.latex', preamble='\usepackage{color}')
matplotlibrc('text',usetex=True)
matplotlibrc('font', family='serif')
figure_folder = "../report/"

#Final solution definition for comparison in problem 2
def solution(c1,c2,m,g1,g2,X,Tf,alpha,c,Lx):
    sol1 = np.exp(-Tf*(2*np.pi/Lx)**2*alpha)*np.sin((2*np.pi*(X-c*Tf)/Lx)-g1)
    sol2 = np.exp(-Tf*(2*m*np.pi/Lx)**2*alpha)*np.cos((2*m*np.pi*(X-c*Tf)/Lx)-g2)
    sol = c1*sol1 - c2*sol2
    return sol    

#Flags for each problem    
Problem_1 = True
Problem_2 = True
Problem_3 = True
if Problem_1:
    
    #Definition of numbr of iterations
    N = np.logspace(0,3,num=1000)
    
    #Define truncation error arrays
    e_TR_16 = np.zeros(N.size,dtype=np.float64)
    e_TR_32 = np.zeros(N.size,dtype=np.float64)
    e_TR_64 = np.zeros(N.size,dtype=np.float64)
    
    #float16
    p1 = np.float16(np.pi)
    P1 = np.pi*np.ones(1, dtype=np.float16)
    
    #float 32
    p2 = np.float32(np.pi)
    P2 = np.pi*np.ones(1, dtype=np.float32)
    
    #float 64
    p3 = np.float64(np.pi)
    P3 = np.pi*np.ones(1, dtype=np.float64)
    
    for i,n in enumerate(N):
        
        for j in np.arange(1,n+1,1):
            P1 = P1*p1
            P2 = P2*p2
            P3 = P3*p3
        for j in np.arange(1,n+1,1):
            P1 = P1/p1
            P2 = P2/p2
            P3 = P3/p3
            
        e_TR_16[i] = abs(np.pi-np.float64(P1))
        e_TR_32[i] = abs(np.pi-np.float64(P2))
        e_TR_64[i] = abs(np.pi-np.float64(P3))
        
    #Plotting block
    figure_name = "Problem_1.pdf"
    figure_file_path1 = figure_folder + figure_name

    figwidth       = 10
    figheight      = 8
    lineWidth      = 3
    textFontSize   = 10
    gcafontSize    = 14


    fig = plt.figure(0, figsize=(figwidth,figheight))

    ax_1 = fig.add_subplot(111)
    
    ax = ax_1
    plt.axes(ax)
    ax.loglog(N,e_TR_16,'-r',linewidth=lineWidth,label="float16")
    ax.loglog(N,e_TR_32,'-b',linewidth=lineWidth,label="float32")
    ax.loglog(N,e_TR_64,'-g',linewidth=lineWidth,label="float64")
#    ax.loglog(N,dx**3, '--')
#    ax.loglog(dx_inv,dx**4, '--')
#    ax.loglog(dx_inv,dx**5, '--')
    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
    ax.grid('on',which='both')

    ax.set_xlabel("$n$",fontsize=textFontSize+2)
    ax.set_ylabel(r"$epsilon_{TR}$",fontsize=textFontSize+2,rotation=90)
    plt.title("Truncation Error vs $n$",fontsize=textFontSize+4)
    plt.legend(loc='best')
    print "Saving Figure:" + figure_file_path1
    plt.savefig(figure_file_path1)



if Problem_2:
   
   #Part b
   
   #Define solution arrays
   U1 = np.zeros([6,10])
   U2 = np.zeros([6,25])
   U3 = np.zeros([6,50])
   U4 = np.zeros([6,100])
   
   #Parameters given as input to the solver
   diff_const = 0.5
   w_speed = 2.0
   cfl = 0.1
   l = 1.0
   c1 = 2.0
   c2 = 2.0
   m = 2.0
   g1 = 2.0
   g2 = 2.0
   final_t = 1.0/(diff_const*(2*np.pi*m/l)**2)
   
   #Evaluate the solution for Np=10,25,50 and 100
   
   #Np=10
   A1, B1, U1[0,:] = solve.solver(10,w_speed,diff_const,cfl,final_t,l,"Explicit-Euler","1st-order-upwind",0,1)
   A2, B2, U1[1,:] = solve.solver(10,w_speed,diff_const,cfl,final_t,l,"Explicit-Euler","2nd-order-upwind",0,1)
   A3, B3, U1[2,:] = solve.solver(10,w_speed,diff_const,cfl,final_t,l,"Explicit-Euler","2nd-order-central",0,1)
   A4, B4, U1[3,:] = solve.solver(10,w_speed,diff_const,cfl,final_t,l,"Crank-Nicolson","1st-order-upwind",0,1)
   A5, B5, U1[4,:] = solve.solver(10,w_speed,diff_const,cfl,final_t,l,"Crank-Nicolson","2nd-order-upwind",0,1)
   A6, B6, U1[5,:] = solve.solver(10,w_speed,diff_const,cfl,final_t,l,"Crank-Nicolson","2nd-order-central",0,1)
   xx = np.linspace(0.,l,11)
   x_mesh1 = 0.5*(xx[:-1]+xx[1:])
   
   #Np=25
   A7, B7, U2[0,:] = solve.solver(25,w_speed,diff_const,cfl,final_t,l,"Explicit-Euler","1st-order-upwind",0,1)
   A8, B8, U2[1,:] = solve.solver(25,w_speed,diff_const,cfl,final_t,l,"Explicit-Euler","2nd-order-upwind",0,1)
   A9, B9, U2[2,:] = solve.solver(25,w_speed,diff_const,cfl,final_t,l,"Explicit-Euler","2nd-order-central",0,1)
   A10, B10, U2[3,:] = solve.solver(25,w_speed,diff_const,cfl,final_t,l,"Crank-Nicolson","1st-order-upwind",0,1)
   A11, B11, U2[4,:] = solve.solver(25,w_speed,diff_const,cfl,final_t,l,"Crank-Nicolson","2nd-order-upwind",0,1)
   A12, B12, U2[5,:] = solve.solver(25,w_speed,diff_const,cfl,final_t,l,"Crank-Nicolson","2nd-order-central",0,1)
   xx = np.linspace(0.,l,26)
   x_mesh2 = 0.5*(xx[:-1]+xx[1:])
   
   #Np=50
   A13, B13, U3[0,:] = solve.solver(50,w_speed,diff_const,cfl,final_t,l,"Explicit-Euler","1st-order-upwind",0,1)
   A14, B14, U3[1,:] = solve.solver(50,w_speed,diff_const,cfl,final_t,l,"Explicit-Euler","2nd-order-upwind",0,1)
   A15, B15, U3[2,:] = solve.solver(50,w_speed,diff_const,cfl,final_t,l,"Explicit-Euler","2nd-order-central",0,1)
   A16, B16, U3[3,:] = solve.solver(50,w_speed,diff_const,cfl,final_t,l,"Crank-Nicolson","1st-order-upwind",0,1)
   A17, B17, U3[4,:] = solve.solver(50,w_speed,diff_const,cfl,final_t,l,"Crank-Nicolson","2nd-order-upwind",0,1)
   A18, B18, U3[5,:] = solve.solver(50,w_speed,diff_const,cfl,final_t,l,"Crank-Nicolson","2nd-order-central",0,1)
   xx = np.linspace(0.,l,51)
   x_mesh3 = 0.5*(xx[:-1]+xx[1:])
   
   #Np=100
   A19, B19, U4[0,:] = solve.solver(100,w_speed,diff_const,cfl,final_t,l,"Explicit-Euler","1st-order-upwind",0,1)
   A20, B20, U4[1,:] = solve.solver(100,w_speed,diff_const,cfl,final_t,l,"Explicit-Euler","2nd-order-upwind",0,1)
   A21, B21, U4[2,:] = solve.solver(100,w_speed,diff_const,cfl,final_t,l,"Explicit-Euler","2nd-order-central",0,1)
   A22, B22, U4[3,:] = solve.solver(100,w_speed,diff_const,cfl,final_t,l,"Crank-Nicolson","1st-order-upwind",0,1)
   A23, B23, U4[4,:] = solve.solver(100,w_speed,diff_const,cfl,final_t,l,"Crank-Nicolson","2nd-order-upwind",0,1)
   A24, B24, U4[5,:] = solve.solver(100,w_speed,diff_const,cfl,final_t,l,"Crank-Nicolson","2nd-order-central",0,1)
   xx = np.linspace(0.,l,101)
   x_mesh4 = 0.5*(xx[:-1]+xx[1:])
   
   #Analytical solution at final time
   U = solution(c1,c2,m,g1,g2,x_mesh4,final_t,diff_const,w_speed,l)
   
   #Plotting block
   figure_name = "Problem2_Part_b.pdf"
   figure_file_path2b = figure_folder + figure_name
    
   figwidth       = 40
   figheight      = 40
   lineWidth      = 3
   textFontSize   = 14
   gcafontSize    = 14
    
    
   fig = plt.figure(0, figsize=(figwidth,figheight))
    
   ax_2 = fig.add_subplot(221)
    
   ax = ax_2
   plt.axes(ax)
   plt.plot(x_mesh1,U1[0,:],'-r',linewidth=lineWidth,label="EE and 1st Order Upwind")
   plt.plot(x_mesh1,U1[1,:],'-k',linewidth=lineWidth,label="EE and 2nd Order Upwind")
   plt.plot(x_mesh1,U1[2,:],'-b',linewidth=lineWidth,label="EE and 2nd Order Central")
   plt.plot(x_mesh1,U1[3,:],'-g',linewidth=lineWidth,label="CN and 1st Order Upwind")
   plt.plot(x_mesh1,U1[4,:],'-c',linewidth=lineWidth,label="CN and 2nd Order Upwind")
   plt.plot(x_mesh1,U1[5,:],'-y',linewidth=lineWidth,label="CN and 2nd Order Central")
   plt.plot(x_mesh4,U,'--k',linewidth=lineWidth,label="Analytical solution")
   plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
   plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
   ax.grid('on',which='both')
   ax.set_xlabel("x",fontsize=textFontSize)
   ax.set_ylabel(r"u",fontsize=textFontSize,rotation=90)
   plt.title("N=10",fontsize=textFontSize)
   plt.legend(loc='best',fontsize=6)
   
   ax_2 = 0
   ax_2 = fig.add_subplot(222)
    
   ax = ax_2
   plt.axes(ax)
   plt.plot(x_mesh2,U2[0,:],'-r',linewidth=lineWidth,label="EE and 1st Order Upwind")
   plt.plot(x_mesh2,U2[1,:],'-k',linewidth=lineWidth,label="EE and 2nd Order Upwind")
   plt.plot(x_mesh2,U2[2,:],'-b',linewidth=lineWidth,label="EE and 2nd Order Central")
   plt.plot(x_mesh2,U2[3,:],'-g',linewidth=lineWidth,label="CN and 1st Order Upwind")
   plt.plot(x_mesh2,U2[4,:],'-c',linewidth=lineWidth,label="CN and 2nd Order Upwind")
   plt.plot(x_mesh2,U2[5,:],'-y',linewidth=lineWidth,label="CN and 2nd Order Central")
   plt.plot(x_mesh4,U,'--k',linewidth=lineWidth,label="Analytical solution")
   plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
   plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
   ax.grid('on',which='both')
   ax.set_xlabel("x",fontsize=textFontSize)
   ax.set_ylabel(r"u",fontsize=textFontSize,rotation=90)
   plt.title("N=25",fontsize=textFontSize)
   plt.legend(loc='best',fontsize=6)
   
   ax_2 = 0
   ax_2 = fig.add_subplot(223)
    
   ax = ax_2
   plt.axes(ax)
   plt.plot(x_mesh3,U3[0,:],'-r',linewidth=lineWidth,label="EE and 1st Order Upwind")
   plt.plot(x_mesh3,U3[1,:],'-k',linewidth=lineWidth,label="EE and 2nd Order Upwind")
   plt.plot(x_mesh3,U3[2,:],'-b',linewidth=lineWidth,label="EE and 2nd Order Central")
   plt.plot(x_mesh3,U3[3,:],'-g',linewidth=lineWidth,label="CN and 1st Order Upwind")
   plt.plot(x_mesh3,U3[4,:],'-c',linewidth=lineWidth,label="CN and 2nd Order Upwind")
   plt.plot(x_mesh3,U3[5,:],'-y',linewidth=lineWidth,label="CN and 2nd Order Central")
   plt.plot(x_mesh4,U,'--k',linewidth=lineWidth,label="Analytical solution")
   plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
   plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
   ax.grid('on',which='both')
   ax.set_xlabel("x",fontsize=textFontSize)
   ax.set_ylabel(r"u",fontsize=textFontSize,rotation=90)
   plt.title("N=50",fontsize=textFontSize)
   plt.legend(loc='best',fontsize=6)
   
   ax_2 = 0
   ax_2 = fig.add_subplot(224)
    
   ax = ax_2
   plt.axes(ax)
   plt.plot(x_mesh4,U4[0,:],'-r',linewidth=lineWidth,label="EE and 1st Order Upwind")
   plt.plot(x_mesh4,U4[1,:],'-k',linewidth=lineWidth,label="EE and 2nd Order Upwind")
   plt.plot(x_mesh4,U4[2,:],'-b',linewidth=lineWidth,label="EE and 2nd Order Central")
   plt.plot(x_mesh4,U4[3,:],'-g',linewidth=lineWidth,label="CN and 1st Order Upwind")
   plt.plot(x_mesh4,U4[4,:],'-c',linewidth=lineWidth,label="CN and 2nd Order Upwind")
   plt.plot(x_mesh4,U4[5,:],'-y',linewidth=lineWidth,label="CN and 2nd Order Central")
   plt.plot(x_mesh4,U,'--k',linewidth=lineWidth,label="Analytical solution")
   plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
   plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
   ax.grid('on',which='both')
   ax.set_xlabel("x",fontsize=textFontSize)
   ax.set_ylabel(r"u",fontsize=textFontSize,rotation=90)
   plt.title("N=100",fontsize=textFontSize)
   plt.legend(loc='best',fontsize=6)
   print "Saving Figure:" + figure_file_path2b
   plt.savefig(figure_file_path2b)
   
   #Part C
   
   #Fixed dt
   
   #No. of points and grid spacing
   N = np.logspace(1,2,num=20,base=10)
   Dx_inv = (N-1)/l
   
   #truncation error
   e_rms1 = np.zeros([3,N.size])
   
   #dt
   time_step = 0.00001
   
   #evaluate solution for each N using Crank Nicolson and 3 spatial discretization schemes for advection
   for i,n in enumerate(N):
      u1 = np.zeros([3,int(n)])
      a, b, u1[0,:] = solve.solver(n,w_speed,diff_const,cfl,final_t,l,"Crank-Nicolson","1st-order-upwind",1,time_step)
      a, b, u1[1,:] = solve.solver(n,w_speed,diff_const,cfl,final_t,l,"Crank-Nicolson","2nd-order-upwind",1,time_step)
      a, b, u1[2,:] = solve.solver(n,w_speed,diff_const,cfl,final_t,l,"Crank-Nicolson","2nd-order-central",1,time_step)
      answ1 = np.zeros(int(n))
      
      #mesh definition and staggering
      x_x = np.linspace(0.,l,n+1)
      mesh = 0.5*(x_x[:-1]+x_x[1:])
      
      #analytical solution
      answ1 = solution(c1,c2,m,g1,g2,mesh,final_t,diff_const,w_speed,l)
      
      #evaluate RMS error for each spatial discretization
      e_rms1[0,i] = np.linalg.norm(u1[0,:]-answ1)/np.sqrt(n)
      e_rms1[1,i] = np.linalg.norm(u1[1,:]-answ1)/np.sqrt(n)
      e_rms1[2,i] = np.linalg.norm(u1[2,:]-answ1)/np.sqrt(n)
    
   #fixed dx
    
   #Define array for dt
   time = np.logspace(-1,-4,num=40,base=10)
   
   #Fixed Np and hence dx
   Np = 100
   u2 = np.zeros([3,Np])
   
   #Define RMS error array
   e_rms2 = np.zeros([3,time.size])
   
   #mesh defn. and staggering
   x_x = np.linspace(0.,l,Np+1)
   mesh = 0.5*(x_x[:-1]+x_x[1:])
   
   #analytical solution
   answ2 = solution(c1,c2,m,g1,g2,mesh,final_t,diff_const,w_speed,l)
   
   #solution for each time step size using Crank Nicolson and 3 spatial discretizations for advection
   for i,Dt in enumerate(time):
      a, b, u2[0,:] = solve.solver(Np,w_speed,diff_const,cfl,final_t,l,"Crank-Nicolson","1st-order-upwind",1,Dt)
      a, b, u2[1,:] = solve.solver(Np,w_speed,diff_const,cfl,final_t,l,"Crank-Nicolson","2nd-order-upwind",1,Dt)
      a, b, u2[2,:] = solve.solver(Np,w_speed,diff_const,cfl,final_t,l,"Crank-Nicolson","2nd-order-central",1,Dt)
      e_rms2[0,i] = np.linalg.norm(u2[0,:]-answ2)*Dt/(np.sqrt(Np)*final_t)
      e_rms2[1,i] = np.linalg.norm(u2[1,:]-answ2)*Dt/(np.sqrt(Np)*final_t)
      e_rms2[2,i] = np.linalg.norm(u2[2,:]-answ2)*Dt/(np.sqrt(Np)*final_t)
      
   figure_name = "Problem2_Part_c.pdf"
   figure_file_path2c = figure_folder + figure_name
    
   figwidth       = 40
   figheight      = 40
   lineWidth      = 3
   textFontSize   = 14
   gcafontSize    = 14
    
   fig = plt.figure(1, figsize=(figwidth,figheight))
   ax_2 = 0
   ax_2 = fig.add_subplot(2,1,1)
   
   ax = ax_2
   plt.axes(ax)
   ax.loglog(Dx_inv,e_rms1[0,:],'-k',linewidth=lineWidth,label="RMS Error-1st-0rder-upwind")
   ax.loglog(Dx_inv,e_rms1[1,:],'-r',linewidth=lineWidth,label="RMS Error-2nd-0rder-upwind")
   ax.loglog(Dx_inv,e_rms1[2,:],'-b',linewidth=lineWidth,label="RMS Error-2nd-0rder-central")
   ax.loglog(Dx_inv,Dx_inv**-1,'m--',label="Order-1")
   ax.loglog(Dx_inv,Dx_inv**-2,'r--',label="Order-2")
   plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
   plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
   ax.grid('on',which='both')
  #  ax.set_xticks()
   ax.set_xlim(left=10)
  #  ax.set_yticks()
  #  ax.set_ylim()
   ax.set_xlabel(r"$dx^{-1}$",fontsize=textFontSize)
   ax.set_ylabel(r"$epsilon_{RMS}$",fontsize=textFontSize,rotation=90)
   plt.title("Fixed dt")
   plt.legend(loc='best')
   
   ax_2 = 0
   ax_2 = fig.add_subplot(2,1,2)
   ax = ax_2
   plt.axes(ax)
   ax.loglog(time**-1,e_rms2[0,:],'-k',linewidth=lineWidth,label="Scaled RMS Error-1st-0rder-upwind")
   ax.loglog(time**-1,e_rms2[1,:],'-r',linewidth=lineWidth,label="Scaled Error-2nd-0rder-upwind")
   ax.loglog(time**-1,e_rms2[2,:],'-b',linewidth=lineWidth,label="Scaled Error-2nd-0rder-central")
   ax.loglog(time**-1,time,'m--',label="Order-1")
   ax.loglog(time**-1,time**2,'r--',label="Order-2")
#   ax.loglog(dx_inverse,dx_range,'m--')
#   ax.loglog(dx_inverse,dx_range**2,'r--')
#   ax.loglog(dx_inverse,dx_range**3,'g--')
#   ax.loglog(dx_inverse,dx_range**4,'b--')
#   ax.loglog(dx_inverse,dx_range**5,'k--')
#   ax.loglog(dx_inverse,dx_range**6,'c--')
   plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
   plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
   ax.grid('on',which='both')
  #  ax.set_xticks()
  #  ax.set_xlim()
  #  ax.set_yticks()
  #  ax.set_ylim()
   ax.set_xlabel(r"$dt^{-1}$",fontsize=textFontSize)
   ax.set_ylabel(r"Scaled $epsilon_{RMS}$",fontsize=textFontSize,rotation=90)
   plt.title("Fixed dx")
   plt.legend(loc='best')
   
   
   print "Saving Figure:" + figure_file_path2c
   plt.savefig(figure_file_path2c)
   
   
   #Part D: Spy plots of A and B for Np=10
   figure_name = "Problem2_Part_d_A.pdf"
   figure_file_path2d = figure_folder + figure_name
    
   figwidth       = 30
   figheight      = 40
   lineWidth      = 3
   textFontSize   = 14
   gcafontSize    = 14
    
    
   fig = plt.figure(2, figsize=(figwidth,figheight))
   
   ax_2 = 0
   ax_2 = fig.add_subplot(1,6,1) 
   ax_2.spy(A1)
   plt.xlabel("EE and 1st order upwind",fontsize=8)
   ax_2 = fig.add_subplot(1,6,2) 
   ax_2.spy(A2)
   plt.xlabel("EE and 2nd order upwind",fontsize=8)
   ax_2 = 0
   ax_2 = fig.add_subplot(1,6,3) 
   ax_2.spy(A3)
   plt.xlabel("EE and 2nd order central",fontsize=8)
   ax_2 = 0
   ax_2 = fig.add_subplot(1,6,4) 
   ax_2.spy(A4)
   plt.xlabel("CN and 1st order upwind",fontsize=8)
   ax_2 = 0
   ax_2 = fig.add_subplot(1,6,5) 
   ax_2.spy(A5)
   plt.xlabel("CN and 2nd order upwind",fontsize=8)
   ax_2 = 0
   ax_2 = fig.add_subplot(1,6,6) 
   ax_2.spy(A6)
   plt.xlabel("CN and 2nd order central",fontsize=8)
   
   print "Saving Figure:" + figure_file_path2d
   plt.savefig(figure_file_path2d)
   
   figure_name = "Problem2_Part_d_B.pdf"
   figure_file_path2d_1 = figure_folder + figure_name
   
   fig2 = plt.figure(5, figsize=(figwidth,figheight))
   
   ax_2 = 0
   ax_2 = fig2.add_subplot(1,6,1)
   ax_2.spy(B1)
   plt.xlabel("EE and 1st order upwind",fontsize=8)
   ax_2 = 0
   ax_2 = fig2.add_subplot(1,6,2)
   ax_2.spy(B2)
   plt.xlabel("EE and 2nd order upwind",fontsize=8)
   ax_2 = 0
   ax_2 = fig2.add_subplot(1,6,3)
   ax_2.spy(B3)
   plt.xlabel("EE and 2nd order central",fontsize=8)
   ax_2 = 0
   ax_2 = fig2.add_subplot(1,6,4)
   ax_2.spy(B4)
   plt.xlabel("CN and 1st order upwind",fontsize=8)
   ax_2 = 0
   ax_2 = fig2.add_subplot(1,6,5)
   ax_2.spy(B5)
   plt.xlabel("CN and 1st order upwind",fontsize=8)
   ax_2 = 0
   ax_2 = fig2.add_subplot(1,6,6)
   ax_2.spy(B6)
   plt.xlabel("CN and 1st order central",fontsize=8)
   
   print "Saving Figure:" + figure_file_path2d_1
   plt.savefig(figure_file_path2d_1)
   
   # Part E
   C_c = np.linspace(0,2.0,20)
   C_alpha = np.linspace(0,2.0,20)
   dt = 0.001
   Np = 100
   dx = l/(Np)
   Alpha = C_alpha*dx*dx/dt
   C = C_c*dx/dt
   u_e = np.zeros([3,int(Np)])
   lambda1 = np.zeros([C_c.size,C_alpha.size])
   lambda2 = np.zeros([C_c.size,C_alpha.size])
   lambda3 = np.zeros([C_c.size,C_alpha.size])
   for i,alpha in enumerate(Alpha):
       for j,c in enumerate(C):
           a1, b1, u_e[0,:] = solve.solver(Np,c,alpha,cfl,final_t,l,"Explicit-Euler","1st-order-upwind",1,dt)
           a2, b2, u_e[1,:] = solve.solver(Np,c,alpha,cfl,final_t,l,"Explicit-Euler","2nd-order-upwind",1,dt)
           a3, b3, u_e[2,:] = solve.solver(Np,c,alpha,cfl,final_t,l,"Explicit-Euler","2nd-order-central",1,dt)
           T1 = scylinalg.inv(a1).dot(b1)
           T2 = scylinalg.inv(a2).dot(b2)
           T3 = scylinalg.inv(a3).dot(b3)
           lambdas1,_= scylinalg.eig(T1)
           lambdas2,_ = scylinalg.eig(T2)
           lambdas3,_ = scylinalg.eig(T3)
           lambda1[i,j] = np.max(np.abs(lambdas1))
           lambda2[i,j] = np.max(np.abs(lambdas2))
           lambda3[i,j] = np.max(np.abs(lambdas3))
           
   figure_name = "Problem2_Part_e.pdf"
   figure_file_path2e = figure_folder + figure_name
    
   figwidth       = 5
   figheight      = 30
   lineWidth      = 3
   textFontSize   = 10
   gcafontSize    = 14
    
   fig = plt.figure(10, figsize=(figwidth,figheight))
   ax_3 = fig.add_subplot(311)
   CS = ax_3.contourf(C_c,C_alpha,lambda1,levels=[0,2],cmap=plt.cm.coolwarm)
   cbar = plt.colorbar(CS)
   ax_3.set_xlabel(r"$C_c$",fontsize=textFontSize,ha = 'right', va = 'top')
   ax_3.set_ylabel(r"$C_\alpha$",fontsize=textFontSize,rotation=90)
   plt.clabel(CS, inline=1, fontsize=14)
   ax_3 = fig.add_subplot(312)
   CS = ax_3.contourf(C_c,C_alpha,lambda2,levels=[0,2],cmap=plt.cm.gist_rainbow_r)
   cbar = plt.colorbar(CS)
   ax_3.set_xlabel(r"$C_c$",fontsize=textFontSize)
   ax_3.set_ylabel(r"$C_\alpha$",fontsize=textFontSize,rotation=90)
   plt.clabel(CS, inline=1, fontsize=14)
   ax_3 = fig.add_subplot(313)
   CS = ax_3.contourf(C_c,C_alpha,lambda3,levels=[0,2],cmap=plt.cm.brg)
   cbar = plt.colorbar(CS)
   ax_3.set_xlabel(r"$C_c$",fontsize=textFontSize)
   ax_3.set_ylabel(r"$C_\alpha$",fontsize=textFontSize,rotation=90)
   plt.clabel(CS, inline=1, fontsize=14)           
   print "Saving Figure:" + figure_file_path2e
   plt.savefig(figure_file_path2e)
   
   
   
if Problem_3:
    
    # solver2(L,Np,c,alpha,beta,dt,a,w,Tf,flag,plot_every) Use this defn to get final solution u
    
    
    #Part a
    L = 10.0
    Np = 100
    c = 2.0
    alpha = 1.0
    beta = 50.0
    dt = 0.005
    a = 1.0
    w = 20.0
    Tf = 10.0
    plot_every = 200
    string = "Problem3_part_a.pdf"
    string2 = "Transient Solution for $T_f = 10$"
    
    u, mesh, q = solve.solver2(L,Np,c,alpha,beta,dt,a,w,Tf,1,plot_every,string,string2)
    
    
    #Commented part to see effect of various parameters on the final solution shape.
    #Change parameters as needed
#    
#    L = 10.0
#    Np = 100
#    c = 2.0
#    alpha = 1.0
#    beta = 10.0
#    dt = 0.005
#    a = 1.0
#    w = 20.0
#    Tf = 10.0
#    plot_every = 200
#    string = "Problem3_part_a_Variation_Plots.pdf"
#    string2 = "Transient Solution for $T_f = 10$ changing one or more parameter"
#    
#    u, mesh, q = solve.solver2(L,Np,c,alpha,beta,dt,a,w,Tf,1,plot_every,string,string2)
    
    
    

    
    
    #Part b
    # Note that boundary layer thickness changes slightly for higher time steps
    #this is due to the slight distortion of the solution because of higher CFL numbers
    #Preferably, keep dt<0.04
    
    #pi1 vs pi4 keeping pi5 and pi3 constant (for defn of pi grps, see report)
    L = 10.0
    Np = 100
    c = 2.0
    alpha = 1.0
    beta = np.linspace(10.0,50.0,5)
    dt = 0.001
    a = 1.0
    w = 20.0
    delta = np.zeros(beta.size)
    pi4 = beta/w
    Tf = 10.0
    plot_every = 200
    
    figure_name = "Problem_3_partb.pdf"
    figure_file_path3 = figure_folder + figure_name

    figwidth       = 10
    figheight      = 8
    lineWidth      = 3
    textFontSize   = 10
    gcafontSize    = 14
    
    fig = plt.figure(4, figsize=(figwidth,figheight))
    
    for i,Beta in enumerate(beta):
        u, mesh, q = solve.solver2(L,Np,c,alpha,Beta,dt,a,w,Tf,0,plot_every,string,string2)
        delta[i] = solve.delta(u,mesh,q)
    
    pi1 = delta*w/c    
    ax_5 = fig.add_subplot(211)
    plt.axes(ax_5)
    ax_5.plot(pi4,pi1,'r^',linewidth=lineWidth)
    plt.setp(ax_5.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax_5.get_yticklabels(),fontsize=gcafontSize)
    ax_5.grid('on',which='both')

    ax_5.set_xlabel("$\Pi_4$",fontsize=textFontSize+2)
    ax_5.set_ylabel(r"$\Pi_1}$",fontsize=textFontSize+2,rotation=90)
    plt.title("Variation of $\Pi_1$ against $\Pi_4$",fontsize=textFontSize+4)
        
    
    #pi1 vs pi5 keeping pi4 and pi3 constant (for defn of pi grps, see report)
    L = 10.0
    Np = 100
    c = 2.0
    alpha = np.linspace(0.5,2.5,5)
    beta = 20.0
    dt = 0.001
    a = 1.0
    w = 20.0
    delta = np.zeros(alpha.size)
    pi3 = alpha*w/c**2
    Tf = 10.0
    plot_every = 200
    
    for i,a in enumerate(alpha):
        u, mesh, q = solve.solver2(L,Np,c,a,beta,dt,a,w,Tf,0,plot_every,string,string2)
        delta[i] = solve.delta(u,mesh,q)
    
    pi1 = delta*w/c
    ax_5 = fig.add_subplot(212)
    plt.axes(ax_5)
    ax_5.plot(pi3,pi1,'r^',linewidth=lineWidth)
    plt.setp(ax_5.get_xticklabels(),fontsize=gcafontSize)
    plt.setp(ax_5.get_yticklabels(),fontsize=gcafontSize)
    ax_5.grid('on',which='both')
    

    ax_5.set_xlabel("$\Pi_3$",fontsize=textFontSize+2)
    ax_5.set_ylabel(r"$\Pi_1$",fontsize=textFontSize+2,rotation=90)
    plt.title("Variation of $\Pi_1$ against $\Pi_3$",fontsize=textFontSize+4)
    plt.tight_layout()
    
    print "Saving Figure:" + figure_file_path3
    plt.savefig(figure_file_path3)

    
    
    
    
