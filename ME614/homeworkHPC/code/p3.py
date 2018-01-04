import os
import sys
import numpy as np   # library to handle arrays like Matlab
import scipy.sparse as scysparse
from matplotlib import pyplot as plt
from mpi4py import MPI
import pickle

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size1 = comm.Get_size()


N = ((10**5)+1)*np.ones(1,dtype=np.float128)

e_tr = np.zeros(1,dtype=np.float128)

def function(x):
    f = np.tanh(x,dtype=np.float128)*np.exp(x,dtype=np.float128)
    return f
     

    
dx = np.float128(20.0/(N-1))
    
#Try and avoid defining variables in terms of one another!!!!
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

for i,x in enumerate(x_stencil):
        p_integral = p_integral + (dx*(function(x) + (4*function(x+dx/2)) + function(x+dx))/6)
    
    
comm.Reduce(p_integral, integral, op=MPI.SUM, root=0)

if rank == 0 :
	integral_a = (-1/np.exp(10,dtype=np.float128)) + (np.exp(10,dtype=np.float128)) + (2*np.arctan(1/np.exp(10,dtype=np.float128),dtype=np.float128)) - (2*np.arctan(np.exp(10,dtype=np.float128),dtype=np.float128))
        e_tr = np.abs(integral_a - integral)
        print "The numerical value of the integral is %2.11f" % (integral)
        print "The analytical value of the integral is %2.11f" % (integral_a)
        print "The absoulte truncation error is %2.11f" % (e_tr)
	print "The number of processors used is %d" % (size1)
comm.Barrier()
if rank == 0: 
    f1 = open("Truncation_Error_partc_Np=%d.p" % (size1), "wb")
    pickle.dump(e_tr, f1)
    f1.close()

comm.Barrier()
print "Node %d exiting\n" % (rank)
sys.exit()
