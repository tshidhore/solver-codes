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
import pylab as plt
from matplotlib import rc as matplotlibrc

machine_epsilon = np.finfo(float).eps
figure_folder = "../report/"

### this example script is hard-coded for periodic problems

#########################################
############### User Input ##############
def solver(Nx,c_x,alpha,CFL,Tf,Lx,time_advancement,advection_scheme,flag,delT):
#Nx  = 64

#CFL = 0.1  # Courant-Friedrichs-Lewy number (i.e. dimensionless time step)
#    c_x_ref = 2.0
#c_x   = 1.*c_x_ref  # (linear) convection speed
#alpha = 0.0      # diffusion coefficients
#    Tf  = Lx/(c_x_ref+machine_epsilon) # one complete cycle

#    plot_every  = 100

## Time Advancement
#time_advancement = "Explicit-Euler"
#time_advancement = "Crank-Nicolson"

## Advection Scheme
#advection_scheme = "1st-order-upwind"
#advection_scheme = "2nd-order-central"
#advection_scheme = "2nd-order-upwind", #QUICK

## Diffusion Scheme
    diffusion_scheme = "2nd-order-central" # always-second-order central

    ymin = 0.
    ymax = 1.

    def u_initial(X):
        c1 = 2.0
        c2 = 2.0
        m = 2.0
        g1 = 2.0
        g2 = 2.0
        return c1*np.sin((2*np.pi*X/Lx)-g1)-c2*np.cos((2*m*np.pi*X/Lx)-g2) # lambda functions are better..

#########################################
######## Preprocessing Stage ############

    xx = np.linspace(0.,Lx,Nx+1)
# actual mesh points are off the boundaries x=0, x=Lx
# non-periodic boundary conditions created with ghost points
    x_mesh = 0.5*(xx[:-1]+xx[1:])
    dx  = np.diff(xx)[0]
    dx2 = dx*dx

# for linear advection/diffusion time step is a function
# of c,alpha,dx only; we use reference limits, ballpark
# estimates for Explicit Euler
    if flag==0:
        dt_max_advective = dx/(c_x+machine_epsilon)             #   think of machine_epsilon as zero
        dt_max_diffusive = dx2/(alpha+machine_epsilon)
        dt_max = np.min([dt_max_advective,dt_max_diffusive])
        dt = CFL*dt_max
    elif flag==1:
        dt = delT
#        print "CFL_adv = %2.3f" %((c_x+machine_epsilon)*dt/dx)
#        print "CFL_diff = %2.3f" %((alpha+machine_epsilon)*dt/dx2)
        
# unitary_float = 1.+0.1*machine_epsilon # wat ?!

# Creating identity matrix
    Ieye = scysparse.identity(Nx)

# Creating first derivative
    Dx = spatial_discretization.Generate_Spatial_Operators(x_mesh,advection_scheme,derivation_order=1)
# Creating second derivative
    D2x2 = spatial_discretization.Generate_Spatial_Operators(x_mesh,diffusion_scheme,derivation_order=2)

# Creating A,B matrices such that: 
#     A*u^{n+1} = B*u^{n} + q
    if time_advancement=="Explicit-Euler":
        A = Ieye
        B = Ieye-dt*c_x*Dx+dt*alpha*D2x2
    if time_advancement=="Crank-Nicolson":
        adv_diff_Op = -dt*c_x*Dx+dt*alpha*D2x2
        A = Ieye-0.5*adv_diff_Op
        B = Ieye+0.5*adv_diff_Op

#plt.spy(Dx)
#plt.show()

# forcing csr ordering..
    A , B = scysparse.csr_matrix(A),scysparse.csr_matrix(B)

#########################################
####### Eigenvalue analysis #############
#T = (scylinalg.inv(A.todense())).dot(B.todense())  # T = A^{-1}*B
#lambdas,_ = scylinalg.eig(T); plt.plot(np.abs(lambdas)); plt.show()
#keyboard()

#########################################
########## Time integration #############

    u = u_initial(x_mesh) # initializing solution

# Figure settings
#matplotlibrc('text.latex', preamble='\usepackage{color}')
#matplotlibrc('text',usetex=True)
#matplotlibrc('font', family='serif')

#    figwidth       = 10
#    figheight      = 6
#    lineWidth      = 4
#    textFontSize   = 28
#    gcafontSize    = 30
#    
#    plt.ion()      # pylab's interactive mode-on
#    plt.close()
    
    time = 0.
    it   = 0

# Plot initial conditions
#    fig = plt.figure(0, figsize=(figwidth,figheight))
#    ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#    plt.axes(ax)
#    ax.plot(x_mesh,u_initial(x_mesh),'--k',linewidth=1)
#    ax.text(0.7,0.9,r"$t="+"%1.5f" %time+"$",fontsize=gcafontSize,transform=ax.transAxes)
#    ax.grid('on',which='both')
#    plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
#    plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
#    ax.set_xlabel(r"$x$",fontsize=textFontSize)
#    ax.set_ylabel(r"$u(x,t)$",fontsize=textFontSize,rotation=90)
#    ax.set_ylim([ymin,ymax])
#    plt.draw()

#    _ = raw_input("Plotting Initial Conditions. Press Key to start time integration")
#    plt.close()

    while time < Tf:
    
       it   += 1
       time += dt
   
   # Update solution
   # solving : A*u^{n+1} = B*u^{n} + q
   # where q is zero for periodic and zero source terms
       u = spysparselinalg.spsolve(A,B.dot(u))
   # this operation is repeated many times.. you should
   # prefactorize 'A' to speed up computation.

#       if ~bool(np.mod(it,plot_every)): # plot every plot_every time steps
##       ax.cla()
#           fig = plt.figure(0, figsize=(figwidth,figheight))
#           ax   = fig.add_axes([0.15,0.15,0.8,0.8])
#           plt.axes(ax)
#           ax.plot(x_mesh,u,'-k',linewidth=lineWidth)
#           ax.plot(x_mesh,u_initial(x_mesh),'--k',linewidth=1)
#           ax.text(0.7,0.9,r"$t="+"%1.5f" %time+"$",fontsize=gcafontSize,transform=ax.transAxes)
#           ax.grid('on',which='both')
#           plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
#           plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
#           ax.set_xlabel(r"$x$",fontsize=textFontSize)
#           ax.set_ylabel(r"$u(x,t)$",fontsize=textFontSize,rotation=90)
#           ax.set_ylim([ymin,ymax])
#           plt.show()
#           _ = raw_input("Pausing \n")
##       sleep(1)
#           plt.close(fig)
#           fig=None
       #plt.draw()
       #       plt.tight_layout()
       #       plt.tight_layout()
#      
#       plt.cla()

#    _ = raw_input("Simulation Finished. Press Enter to continue...")
#    plt.close()

    return(A.todense(),B.todense(),u)

def solver2(L,Np,c,alpha,beta,dt,a,w,Tf,flag,plot_every,string,string2):
    
    #Defining the mesh using X = L*zeta^2
    x = L*np.linspace(0.,1.0,Np+1)**2
    #shifting mesh. note that the two boundary points shift inside by 1/2 on either side
    mesh_temp = 0.5*(x[:-1]+x[1:])
    #left ghost point spacing
    left_spacing = mesh_temp[0]
    #right ghost point spacing
    right_spacing = L - mesh_temp[-1]
    #final mesh incluidng 2 ghost points
    mesh = np.zeros(mesh_temp.size+2)
    mesh[0] = -left_spacing
    mesh[-1] = L + right_spacing
    #core retained 
    mesh[1:-1] = mesh_temp
    
    Ieye = scysparse.identity(mesh.size)
    
    #Defining 1st and 2nd spatial derivative operators using 2nd-order-upwind and central respectively
    Dx = spatial_discretization.Generate_Spatial_Operators2(mesh,"2nd-order-upwind",1)
    D2x2 = spatial_discretization.Generate_Spatial_Operators2(mesh,"2nd-order-central",2)
    
    #Advection diffusion operator for pre-conditioning o f A and B. Note that the beta*dt*Ieye term can be completely transfered to B, but this provides better stability
    adv_diff_Op = -dt*c*Dx+dt*alpha*D2x2-dt*beta*Ieye
    A = Ieye-0.5*adv_diff_Op
    B = Ieye+0.5*adv_diff_Op
    A , B = scysparse.csr_matrix(A),scysparse.csr_matrix(B)
    
    #Implementing boundary condition in rows of A and B
    
    #Grid spacing at the right edge
    dx1 = np.diff(mesh)[-1]
    A[0,0] = 0.5
    A[0,1] = 0.5
    A[-1,-2] = -1/dx1
    A[-1,-1] = 1/dx1
    A[0,2:] = 0
    A[-1,:-2] = 0
    B[0,:] = 0
    B[-1,:] = 0
    
    #Solution, time, iteration initialization
    u = np.zeros(mesh.size)
    q = np.zeros(mesh.size)
    time = 0.
    it   = 0
    
    if flag==1:    #Plot figure only if given the flag
        
        figure_name = "Problem3_part_a.pdf"
        figure_file_path3a = figure_folder + figure_name
    
        figwidth       = 20
        figheight      = 20
        lineWidth      = 3
        textFontSize   = 18
        gcafontSize    = 14
    
    
        fig = plt.figure(3, figsize=(figwidth,figheight))
    
    while time < Tf:
    
       it   += 1
       time += dt
       q[0] = a*np.cos(w*time)
       q[1:] = 0
       C = B.dot(u) + q
       u = spysparselinalg.spsolve(A,C)
       
       if flag==1:
           if it%plot_every==0:
               plt.plot(mesh,u,label='t = %f'%(time))
    if flag==1:
        plt.xlabel(r"$x$",fontsize=textFontSize)
        plt.ylabel(r"$u(x,t)$",fontsize=textFontSize,rotation=90)
        plt.xlim([-0.5,L])
        plt.legend(loc='best')           
        plt.title(string2,fontsize=textFontSize)
   
        print "Saving Figure:" + figure_file_path3a
        plt.savefig(figure_file_path3a)
        
    return (u,mesh,q[0])
    
def delta(u,mesh,q):
    for j,x in reversed(list(enumerate(mesh))):
        if np.abs(u[j]) > 0.01*np.abs(q):
            point = x
            break
        
    return point