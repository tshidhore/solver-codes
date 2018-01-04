#Code for weak scaling part
#Basis same as the 1st code

import os
import sys
import numpy as np   # library to handle arrays like Matlab
import scipy.sparse as scysparse
from matplotlib import pyplot as plt
from mpi4py import MPI
import time
import pickle

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size1 = comm.Get_size()


N = size1*(10**7) + 1

c_time = np.zeros(1,dtype=np.float128)

def function(x):
    f = np.tanh(x,dtype=np.float128)*np.exp(x,dtype=np.float128)
    return f
     

    
dx = np.float128(20.0/(N-1))
p_integral = np.zeros(1,dtype=np.float128)
integral = np.zeros(1,dtype=np.float128)
index_start = int(rank*np.floor(N/size1))
start = -10 + (index_start)*dx
if rank == size1 - 1:
	index_end = int(N - 1)
	end = 10

else:
        index_end = int(index_start + np.floor(N/size1))
	end = -10 + index_end*dx
    
x_stencil = np.arange(start, end, dx)
comm.Barrier()
start_time = time.time()
for i,x in enumerate(x_stencil):
        p_integral = p_integral + (dx*(function(x) + (4*function(x+dx/2)) + function(x+dx))/6)

    
    
comm.Reduce(p_integral, integral, op=MPI.SUM, root=0)
stop_time = time.time() - start_time

if rank == 0 :

	c_time[0] = stop_time
	print "The time required for %d processors is %f" % (size1, c_time[0])
    	f = open("Calculation_Time_Partb_Np=%d.p" % (size1), "wb")
    	pickle.dump(c_time, f)
    	f.close()

comm.Barrier()
print "Node %d exiting\n" % (rank)
sys.exit()
