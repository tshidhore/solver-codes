import os
import sys
import numpy as np   # library to handle arrays like Matlab
import scipy.sparse as scysparse
from matplotlib import pyplot as plt
from mpi4py import MPI
import time
import pickle

#Defining MPI variables
comm = MPI.COMM_WORLD #Used to define message passing protocols
rank = comm.Get_rank() #Processor rank
size1 = comm.Get_size() #Total number of processors

#Defining number of points to be iterated over
N_1 = np.logspace(1, 9, num=9, base=10,dtype=np.float128)
one = np.ones(N_1.size,dtype=np.float128)
N = N_1 + one

#Defining arrays for truncation error, grid spacing and computational time
e_tr = np.zeros(N.size,dtype=np.float128)
Dx = np.zeros(N.size,dtype=np.float128)
c_time = np.zeros(N.size,dtype=np.float128)

#Definition of function to be integrated
def function(x):
    f = np.tanh(x,dtype=np.float128)*np.exp(x,dtype=np.float128)
    return f

#Looping over all points    
for j,n in enumerate(N):
    
#grid spacing for each element in N
    dx = np.float128(20.0/(n-1)) # Try removing np.float
    
    #Try and avoid defining variables in terms of one another!!!! The case was giving issues for 16 processors
    p_integral = np.zeros(1,dtype=np.float128)
    integral = np.zeros(1,dtype=np.float128)
    #Defining start and end point (not included in computations for each individual processor
    index_start = int(rank*np.floor(n/size1))
    start = -10 + (index_start)*dx
    #Special case for last processor: end point should be the last point
    if rank == size1 - 1:
        index_end = int(n - 1)
	end = 10

    else:
        index_end = int(index_start + np.floor(n/size1))
	end = -10 + index_end*dx
    #Define individual stencil for each processor
    x_stencil = np.arange(start, end, dx)
    #All processors synchronize
    comm.Barrier()
    start_time = time.time()
    for i,x in enumerate(x_stencil):
        p_integral = p_integral + (dx*(function(x) + (4*function(x+dx/2)) + function(x+dx))/6)
    
    #Add all results across processors to store them on the rank 0 processor. 
    comm.Reduce(p_integral, integral, op=MPI.SUM, root=0)
    stop_time = time.time() - start_time
    #Everything from now on goes to rank 0 processor
    if rank == 0 :
        integral_a = (-1/np.exp(10,dtype=np.float128)) + (np.exp(10,dtype=np.float128)) + (2*np.arctan(1/np.exp(10,dtype=np.float128),dtype=np.float128)) - (2*np.arctan(np.exp(10,dtype=np.float128),dtype=np.float128))
        e_tr[j] = np.abs(integral_a - integral)
	Dx[j] = dx
	c_time[j] = stop_time
        print "The numerical value of the integral is %2.11f" % (integral)
        print "The analytical value of the integral is %2.11f" % (integral_a)
        print "The absoulte truncation error is %2.11f" % (e_tr[j])
	print "The time required for %d points is %f" % (n, c_time[j])
	print "The grid spacing is %2.11f" % (Dx[j])
comm.Barrier()
#store results into files
if rank == 0: 
    f1 = open("Truncation_Error_Np=%d.p" % (size1), "wb")
    pickle.dump(e_tr, f1)
    f1.close()
    if size1 == 1:
    	f2 = open("Grid_Spacing.p", "wb")
    	pickle.dump(Dx, f2)
    	f2.close()

    f3 = open("Calculation_Time_N=10^8+1_Np=%d.p" % (size1), "wb")
    pickle.dump(c_time[7], f3)
    f3.close()
    
comm.Barrier()
print "Node %d exiting\n" % (rank)
sys.exit()
