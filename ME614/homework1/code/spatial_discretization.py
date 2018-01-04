import numpy as np
import scipy.sparse as scysparse
import sys
from pdb import set_trace as keyboard
from matplotlib import pyplot as plt

def Generate_Weights(x_stencil,x_eval,der_order):

    if x_stencil.ndim>1:
        sys.exit("stencil array is not a 1D numpy array")

    derivation_order = int(der_order) # making sure derivation order is integer
    polynomial_order = len(x_stencil)-1
    
    weights	= np.zeros(x_stencil.shape)
    N		= x_stencil.size

    for ix,x in enumerate(x_stencil):
        base_func = np.zeros(N,)
        base_func[ix] = 1.0
        poly_coefs = np.polyfit(x_stencil,base_func,polynomial_order)
        weights[ix] = np.polyval(np.polyder(poly_coefs,derivation_order),x_eval)

    return weights


############################################################
############################################################


def Generate_Spatial_Operators(x_mesh,order_scheme,der_order):

    N = x_mesh.size

    # you should pre-allocate a sparse matrix predicting already the number of non-zero entries
    A = scysparse.lil_matrix((N, N), dtype=np.float64)

#    if order_scheme == "3rd_order": #Right biased scheme based on 1 point before and 2 points after evaluation point
#
#    for i,x_eval in enumerate(x_mesh):
#        
#            if i==0:    # case for 0th row
#                x_stencil = x_mesh[:4] 
#                A[i,:4] = Generate_Weights(x_stencil, x_eval,der_order)
#            
#            elif i==N-2:   #case for (n-2)th row
#                x_stencil = x_mesh[-4:] 
#                A[i,-4:] = Generate_Weights(x_stencil, x_eval,der_order)
#                
#            elif i==N-1:   #case for (n-1)th row
#                x_stencil = x_mesh[-4:] 
#                A[i,-4:] = Generate_Weights(x_stencil, x_eval,der_order)
#                    
#            else:  # all points in interior
#                x_stencil = x_mesh[i-1:i+3]
#                A[i,i-1:i+3] = Generate_Weights(x_stencil, x_eval,der_order)
       
#Right biased doesnt work for 2b as same stencil is passed irrespective of any modifications. The only way out is to solve a mirrored problem       
       
       
 #      Left biased scheme with 2 points before and one point after evaluation point
                
    for i,x_eval in enumerate(x_mesh):
            if i==0:    # case for 0th row
                x_stencil = x_mesh[:4] 
                A[i,:4] = Generate_Weights(x_stencil, x_eval,der_order)
            
            elif i==1:   #case for 1st row
                x_stencil = x_mesh[:4] 
                A[i,:4] = Generate_Weights(x_stencil, x_eval,der_order)
                
            elif i==N-1:   #case for (n-1)th row
                x_stencil = x_mesh[-4:] 
                A[i,-4:] = Generate_Weights(x_stencil, x_eval,der_order)
        
            else:  # all points in interior
                x_stencil = x_mesh[i-2:i+2]
                A[i,i-2:i+2] = Generate_Weights(x_stencil, x_eval,der_order)
            
    
    if order_scheme == "5th_order":    #Right biased scheme based on 2 point before and 3 points after evaluation point

#For right biased scheme, last 3 rows have same set of stencil points for the polynomial weight generation 
#   Fitting a 5th order polynomial through 6 points
        for i,x_eval in enumerate(x_mesh):
        
            if i==0: # case for 0th row
                x_stencil = x_mesh[:6]
                A[i,:6] = Generate_Weights(x_stencil, x_eval,der_order)
                
            elif i==1:  #case for 1st row
                x_stencil = x_mesh[:6] 
                A[i,:6] = Generate_Weights(x_stencil, x_eval,der_order) 
            
            elif i==N-1: #case for last row
                x_stencil = x_mesh[-6:] 
                A[i,-6:] = Generate_Weights(x_stencil, x_eval,der_order)
                
            elif i==N-2:  #case for 2nd last row
                x_stencil = x_mesh[-6:] 
                A[i,-6:] = Generate_Weights(x_stencil, x_eval,der_order)
                
            elif i==N-3:  #case for 3rd last row
                x_stencil = x_mesh[-6:]
                A[i,-6:] = Generate_Weights(x_stencil, x_eval, der_order)
        
            else:
                x_stencil = x_mesh[i-2:i+4] #case for other rows in the interior
                A[i,i-2:i+4] = Generate_Weights(x_stencil, x_eval,der_order)
                
                
##   Left biased scheme (comment the earlier scheme) based on 3 points before and 2 points after evaluation point            
#                
#                
#            if i==0: # case for 0th row
#                x_stencil = x_mesh[:6]
#                A[i,:6] = Generate_Weights(x_stencil, x_eval,der_order)
#                
#            elif i==1:  #case for 1st row
#                x_stencil = x_mesh[:6] 
#                A[i,:6] = Generate_Weights(x_stencil, x_eval,der_order) 
#            
#            elif i==2: #case for 2nd row
#                x_stencil = x_mesh[:6] 
#                A[i,:6] = Generate_Weights(x_stencil, x_eval,der_order)
#                
#            elif i==N-2:  #case for 2nd last row
#                x_stencil = x_mesh[-6:] 
#                A[i,-6:] = Generate_Weights(x_stencil, x_eval,der_order)
#                
#            elif i==N-1:  #case for last row
#                x_stencil = x_mesh[-6:]
#                A[i,-6:] = Generate_Weights(x_stencil, x_eval, der_order)
#        
#            else:
#                x_stencil = x_mesh[i-3:i+3] #case for other rows in the interior
#                A[i,i-3:i+3] = Generate_Weights(x_stencil, x_eval,der_order)
    
    # convering to csr format, which appears to be more efficient for matrix operations
    return A.tocsc()
