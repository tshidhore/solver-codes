import numpy as np
import scipy.sparse as scysparse
import sys
from pdb import set_trace as keyboard
import scipy.sparse as scysparse
import scipy.sparse.linalg as spysparselinalg
import scipy.linalg as scylinalg        # non-sparse linear algebra
from scipy.integrate import trapz
from scipy.integrate import simps

############################################################
############################################################

def Generate_Weights(x_stencil,x_eval,derivation_order):

    if x_stencil.ndim>1:
        sys.exit("stencil array is not a 1D numpy array")

    derivation_order = int(derivation_order) # making sure derivation order is integer
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

def Generate_Spatial_Operators(x_mesh,scheme,derivation_order):

    N = x_mesh.size
    # you should pre-allocate a sparse matrix predicting already the number of non-zero entries
    circulating_row = np.zeros(N,)
    
    if scheme == "2nd-order-central":
        
        # generating computational molecule
        x_stencil = x_mesh[:3] # first three points
        x_eval    = x_mesh[1]
        weights = Generate_Weights(x_stencil,x_eval,derivation_order)
        circulating_row[-1] = weights[0]
        circulating_row[0]  = weights[1]
        circulating_row[1]  = weights[2]


    if scheme == "2nd-order-upwind": # also called QUICK
    
        # generating computational molecule
        x_stencil = x_mesh[:3] # first three points
        x_eval    = x_mesh[2]
        weights = Generate_Weights(x_stencil,x_eval,derivation_order)
        circulating_row[-2] = weights[0]
        circulating_row[-1] = weights[1]
        circulating_row[0]  = weights[2]

    if scheme == "1st-order-upwind":
        
        # assuming advection velocity is positive, c>0
        # generating computational molecule
        x_stencil = x_mesh[:2] # first two points
        x_eval    = x_mesh[1]
        weights = Generate_Weights(x_stencil,x_eval,derivation_order)
        circulating_row[-1] = weights[0]
        circulating_row[0]  = weights[1]
        
    A_circulant = scylinalg.circulant(circulating_row)
    A           = A_circulant.transpose()
    
    # convering to csr format, which appears to be more efficient for matrix operations
    return scysparse.csr_matrix(A)
    
##################################################################################################################
##################################################################################################################
    
def Generate_Spatial_Operators2(x_mesh,scheme,derivation_order):
    
    N = x_mesh.size
    
    A = scysparse.lil_matrix((N, N), dtype=np.float64)
    
    if scheme == "2nd-order-central":
        for i,x_eval in enumerate(x_mesh):
            if i==0:
                x_stencil = x_mesh[:3]
                A[i,:3] = Generate_Weights(x_stencil,x_eval,derivation_order)
            elif i == N-1:
                x_stencil = x_mesh[-3:]
                A[i,-3:] = Generate_Weights(x_stencil,x_eval,derivation_order)
            else:
                x_stencil = x_mesh[i-1:i+2]
                A[i,i-1:i+2] = Generate_Weights(x_stencil,x_eval,derivation_order)
                
    if scheme == "2nd-order-upwind":
        for i,x_eval in enumerate(x_mesh):
            if i==0:
                x_stencil = x_mesh[:3]
                A[i,:3] = Generate_Weights(x_stencil,x_eval,derivation_order)
            elif i == 1:
                x_stencil = x_mesh[:3]
                A[i,:3] = Generate_Weights(x_stencil,x_eval,derivation_order)
            else:
                x_stencil = x_mesh[i-2:i+1]
                A[i,i-2:i+1] = Generate_Weights(x_stencil,x_eval,derivation_order)
                
    return scysparse.csr_matrix(A)
    
