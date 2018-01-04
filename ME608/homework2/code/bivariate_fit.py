#########################################################
### Written by Matteo Pellegri (ME608 -- Spring 2016) ###
#########################################################

import os
import sys
import numpy as np
import scipy as sp 
import scipy.sparse as scysparse
import scipy.sparse.linalg as splinalg
import pylab as plt
from pdb import set_trace
from pdb import set_trace as keyboard

########################################################
#################### BEGIN FUNCTION ####################
########################################################
def BiVarPolyFit_X(X,Y,x_IN,y_IN,phi_IN,scaling=True):

    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
  
    #Convert x_IN & y_IN to arrays (could possibly be lists)
    x_IN = np.array(x_IN)
    y_IN = np.array(y_IN)

    Npts_IN = x_IN.shape[0]
    Npts_OUT = X.shape[0]
    phi,dphi_dx,d2phi_dx2 = np.zeros(X.shape),np.zeros(X.shape),np.zeros(X.shape)

    if scaling:
        ##### Translation - bringing centroid to 0,0
        x_centroid = np.sum(x_IN)/Npts_IN
        y_centroid = np.sum(y_IN)/Npts_IN
        x_IN = x_IN - x_centroid; X = X - x_centroid
        y_IN = y_IN - y_centroid; Y = Y - y_centroid

        ##### Scaling - dividing by twice the distance of farthest point from origin
        d_IN = np.sqrt(x_IN**2 + y_IN**2)
        d_char = 2.*np.max(d_IN)
        x_IN = x_IN/d_char; X = X/d_char
        y_IN = y_IN/d_char; Y = Y/d_char

    else:
        d_char = 1
    ##### Determine which polynomial to fit based upon the number of input points
    if Npts_IN == 7:
        # f7(x,y) = a(1) + b(X) + c(Y) + d(X^2) + e(Y^2) + f(XY) + g(X^3)
        v = np.array([np.ones(Npts_IN),x_IN,y_IN,x_IN**2,y_IN**2,x_IN*y_IN,x_IN**3])
    elif Npts_IN == 6:
        # f6(x,y) = a(1) + b(X) + c(Y) + d(X^2) + e(Y^2) + f(XY)
        v = np.array([np.ones(Npts_IN),x_IN,y_IN,x_IN**2,y_IN**2,x_IN*y_IN])
    elif Npts_IN == 5:
        # f5(x,y) = a(1) + b(X) + c(Y) + d(X^2) + e(Y^2)
        v = np.array([np.ones(Npts_IN),x_IN,y_IN,x_IN**2,y_IN**2])
    elif Npts_IN == 4:
        # f4(x,y) = a(1) + b(X) + c(Y) + d(X^2)
        v = np.array([np.ones(Npts_IN),x_IN,y_IN,x_IN**2])
    elif Npts_IN == 3:
        # f3(x,y) = a(1) + b(X) + c(Y)
        v = np.array([np.ones(Npts_IN),x_IN,y_IN])
    elif Npts_IN == 8:
        # f7(x,y) = a(1) + b(X) + c(Y) + d(X^2) + e(Y^2) + f(XY) + g(X^3) + h(Y^3)
        v = np.array([np.ones(Npts_IN),x_IN,y_IN,x_IN**2,y_IN**2,x_IN*y_IN,x_IN**3,y_IN**3])
    elif Npts_IN == 9:
        # f7(x,y) = a(1) + b(X) + c(Y) + d(X^2) + e(Y^2) + f(XY) + g(X^3) + h(Y^3) + i(X^2*Y)
        v = np.array([np.ones(Npts_IN),x_IN,y_IN,x_IN**2,y_IN**2,x_IN*y_IN,x_IN**3,y_IN**3,x_IN*x_IN*y_IN])
    elif Npts_IN == 10:
        # f7(x,y) = a(1) + b(X) + c(Y) + d(X^2) + e(Y^2) + f(XY) + g(X^3) + h(Y^3) + i(X^2*Y)
        v = np.array([np.ones(Npts_IN),x_IN,y_IN,x_IN**2,y_IN**2,x_IN*y_IN,x_IN**3,y_IN**3,x_IN*x_IN*y_IN,x_IN*y_IN*y_IN])
    elif Npts_IN == 11:
        # f7(x,y) = a(1) + b(X) + c(Y) + d(X^2) + e(Y^2) + f(XY) + g(X^3) + h(Y^3) + i(X^2*Y)
        v = np.array([np.ones(Npts_IN),x_IN,y_IN,x_IN**2,y_IN**2,x_IN*y_IN,x_IN**3,y_IN**3,x_IN*x_IN*y_IN,x_IN*y_IN*y_IN,x_IN**4])
    else:
        print "###### Error! "
        sys.exit(">>>> Bivariate Fit not implemented for total number of points above 11")

    coeff_der0,_,_,_ = np.linalg.lstsq(v.T, phi_IN)
    coeff_der1x    = np.zeros(coeff_der0.shape)
    coeff_der2x    = np.zeros(coeff_der0.shape)

    if Npts_IN == 7:
        # f7(x,y)  =  a(1)  + b(X) + c(Y) + d(X^2) + e(Y^2) + f(XY) + g(X^3)
        #            [0]     [1]    [2]    [3]      [4]      [5]     [6]

        #            [1]     [3]    [5]     [6]
        # df7/dx   =  b(1) + 2d(X) + f(Y) + 3g(X^2)
        #              [0]     [1]    [2]     [3]
        coeff_der1x[0] = np.array(coeff_der0)[1]
        coeff_der1x[1] = 2 * np.array(coeff_der0)[3]
        coeff_der1x[2] = np.array(coeff_der0)[5]
        coeff_der1x[3] = 3 * np.array(coeff_der0)[6]
        
        #            [3]     [6]
        # d2f7/dx2 = 2d(1) + 6g(X)
        #              [0]     [1]
        coeff_der2x[0] = 2 * np.array(coeff_der0)[3]
        coeff_der2x[1] = 6 * np.array(coeff_der0)[6]

        vv = np.array([np.ones(Npts_OUT),X,Y,X**2,Y**2,X*Y,X**3])

    elif Npts_IN == 8:
        # f7(x,y)  =  a(1)  + b(X) + c(Y) + d(X^2) + e(Y^2) + f(XY) + g(X^3) + g(Y^3)
        #            [0]     [1]    [2]    [3]      [4]      [5]      [6]       [7]

        #            [1]     [3]    [5]     [6]
        # df7/dx   =  b(1) + 2d(X) + f(Y) + 3g(X^2)
        #              [0]     [1]    [2]     [3]
        coeff_der1x[0] = np.array(coeff_der0)[1]
        coeff_der1x[1] = 2 * np.array(coeff_der0)[3]
        coeff_der1x[2] = np.array(coeff_der0)[5]
        coeff_der1x[3] = 3 * np.array(coeff_der0)[6]
        
        #            [3]     [6]
        # d2f7/dx2 = 2d(1) + 6g(X)
        #              [0]     [1]
        coeff_der2x[0] = 2 * np.array(coeff_der0)[3]
        coeff_der2x[1] = 6 * np.array(coeff_der0)[6]

        vv = np.array([np.ones(Npts_OUT),X,Y,X**2,Y**2,X*Y,X**3,Y**3])

    elif Npts_IN == 9:
        # f7(x,y)  =  a(1)  + b(X) + c(Y) + d(X^2) + e(Y^2) + f(XY) + g(X^3) + h(Y^3) + i(X^2Y) 
        #            [0]     [1]    [2]    [3]      [4]      [5]      [6]       [7]     [8]

        #            [1]     [3]    [5]     [6]
        # df7/dx   =  b(1) + 2d(X) + f(Y) + 3g(X^2) + 2i(XY)
        #              [0]     [1]    [2]     [3]
        coeff_der1x[0] = np.array(coeff_der0)[1]
        coeff_der1x[1] = 2 * np.array(coeff_der0)[3]
        coeff_der1x[2] = np.array(coeff_der0)[5]
        coeff_der1x[3] = 3 * np.array(coeff_der0)[6]
        coeff_der1x[4] = 2 * np.array(coeff_der0)[8]
        #            [3]     [6]
        # d2f7/dx2 = 2d(1) + 6g(X) + 2i(Y)
        #              [0]     [1]
        coeff_der2x[0] = 2 * np.array(coeff_der0)[3]
        coeff_der2x[1] = 6 * np.array(coeff_der0)[6]
        coeff_der2x[2] = 2 * np.array(coeff_der0)[8]

        vv = np.array([np.ones(Npts_OUT),X,Y,X**2,Y**2,X*Y,X**3,Y**3,X*X*Y])
    elif Npts_IN == 10:
        # f7(x,y)  =  a(1)  + b(X) + c(Y) + d(X^2) + e(Y^2) + f(XY) + g(X^3) + h(Y^3) + i(X^2Y) +  i(Y^2X)
        #            [0]     [1]    [2]    [3]      [4]      [5]      [6]       [7]     [8]    +    [9]

        #            [1]     [3]    [5]     [6]
        # df7/dx   =  b(1) + 2d(X) + f(Y) + 3g(X^2) + 2i(XY)
        #              [0]     [1]    [2]     [3]
        coeff_der1x[0] = np.array(coeff_der0)[1]
        coeff_der1x[1] = 2 * np.array(coeff_der0)[3]
        coeff_der1x[2] = np.array(coeff_der0)[5]
        coeff_der1x[3] = 3 * np.array(coeff_der0)[6]
        coeff_der1x[4] = 2 * np.array(coeff_der0)[8]
        coeff_der1x[5] = np.array(coeff_der0)[9]
        #            [3]     [6]
        # d2f7/dx2 = 2d(1) + 6g(X) + 2i(Y)
        #              [0]     [1]
        coeff_der2x[0] = 2 * np.array(coeff_der0)[3]
        coeff_der2x[1] = 6 * np.array(coeff_der0)[6]
        coeff_der2x[2] = 2 * np.array(coeff_der0)[8]

        vv = np.array([np.ones(Npts_OUT),X,Y,X**2,Y**2,X*Y,X**3,Y**3,X*X*Y,Y*Y*X])
    elif Npts_IN == 11:
        # f7(x,y)  =  a(1)  + b(X) + c(Y) + d(X^2) + e(Y^2) + f(XY) + g(X^3) + h(Y^3) + i(X^2Y) +  j(Y^2X) + k(X^4)
        #            [0]     [1]    [2]    [3]      [4]      [5]      [6]       [7]     [8]    +    [9]    +    [10]

        #            [1]     [3]    [5]     [6]
        # df7/dx   =  b(1) + 2d(X) + f(Y) + 3g(X^2) + 2i(XY)
        #              [0]     [1]    [2]     [3]
        coeff_der1x[0] = np.array(coeff_der0)[1]
        coeff_der1x[1] = 2 * np.array(coeff_der0)[3]
        coeff_der1x[2] = np.array(coeff_der0)[5]
        coeff_der1x[3] = 3 * np.array(coeff_der0)[6]
        coeff_der1x[4] = 2 * np.array(coeff_der0)[8]
        coeff_der1x[5] = np.array(coeff_der0)[9]
        coeff_der1x[6] = 4 * np.array(coeff_der0)[10]
        #            [3]     [6]
        # d2f7/dx2 = 2d(1) + 6g(X) + 2i(Y)
        #              [0]     [1]
        coeff_der2x[0] = 2 * np.array(coeff_der0)[3]
        coeff_der2x[1] = 6 * np.array(coeff_der0)[6]
        coeff_der2x[2] = 2 * np.array(coeff_der0)[8]
        coeff_der2x[3] = 12 * np.array(coeff_der0)[10]

        vv = np.array([np.ones(Npts_OUT),X,Y,X**2,Y**2,X*Y,X**3,Y**3,X*X*Y,Y*Y*X,X**4])
    elif Npts_IN == 6:
        # f6(x,y)  =  a(1)  + b(X) + c(Y) + d(X^2) + e(Y^2) + f(XY)
        #            [0]     [1]    [2]    [3]      [4]      [5]

        #            [1]     [3]    [5]
        # df6/dx   =  b(1) + 2d(X) + f(Y)
        #              [0]     [1]    [2]
        coeff_der1x[0] = np.array(coeff_der0)[1]
        coeff_der1x[1] = 2 * np.array(coeff_der0)[3]
        coeff_der1x[2] = np.array(coeff_der0)[5]
        
        #            [3]
        # d2f6/dx2 = 2d(1)
        #              [0]
        coeff_der2x[0] = 2 * np.array(coeff_der0)[3]

        vv = np.array([np.ones(Npts_OUT),X,Y,X**2,Y**2,X*Y])

        
    elif Npts_IN == 5:
        # f5(x,y)  =  a(1)  + b(X) + c(Y) + d(X^2) + e(Y^2)
        #            [0]     [1]    [2]    [3]      [4]

        #            [1]     [3]
        # df5/dx   =  b(1) + 2d(X)
        #              [0]     [1]
        coeff_der1x[0] = np.array(coeff_der0)[1]
        coeff_der1x[1] = 2 * np.array(coeff_der0)[3]
        #coeff_der1x[2] = np.array(coeff_der0)[4]

        #            [3]
        # d2f5/dx2 = 2d(1)
        #              [0]
        coeff_der2x[0] = 2 * np.array(coeff_der0)[3]

        vv = np.array([np.ones(Npts_OUT),X,Y,X**2,Y**2])
        
    elif Npts_IN == 4:
        # f4(x,y)  =  a(1)  + b(X) + c(Y) + d(X^2)
        #            [0]     [1]    [2]    [3]
        
        #            [1]     [3]
        # df4/dx   =  b(1) + 2d(X)
        #              [0]     [1]
        coeff_der1x[0] = np.array(coeff_der0)[1]
        coeff_der1x[1] = 2 * np.array(coeff_der0)[3]
        
        #            [3]
        # d2f4/dx2 = 2d(1)
        #              [0]
        coeff_der2x[0] = 2 * np.array(coeff_der0)[3]
        vv = np.array([np.ones(Npts_OUT),X,Y,X**2])
        
    elif Npts_IN == 3:
        # f3(x,y)  = a(1) + b(X) + c(Y)
        #           [0]    [1]    [2]
        
        #           [1]
        # df3/dx   = b(1)
        #             [0]
        coeff_der1x[0] = np.array(coeff_der0)[1]

        # d2f3/dx2 = 0(1)
        #             [0]
        coeff_der2x[0] = 0

        vv = np.array([np.ones(Npts_OUT),X,Y])
  
    phi      = coeff_der0.dot(vv)
    dphi_dx  = coeff_der1x.dot(vv)/d_char
    d2phi_dx2  = coeff_der2x.dot(vv)/(d_char**2)

    return phi,dphi_dx,d2phi_dx2
##################### END FUNCTION #####################

########################################################
#################### BEGIN FUNCTION ####################
########################################################
def BiVarPolyFit_Y(X,Y,x_IN,y_IN,phi_IN,scaling=True):

    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
  
    #Convert x_IN & y_IN to arrays (could possibly be lists)
    x_IN = np.array(x_IN)
    y_IN = np.array(y_IN)

    Npts_IN = x_IN.shape[0]
    Npts_OUT = X.shape[0]
    phi,dphi_dy,d2phi_dy2 = np.zeros(X.shape),np.zeros(X.shape),np.zeros(X.shape)

    if scaling:
        ##### Translation - bringing centroid to 0,0
        x_centroid = np.sum(x_IN)/Npts_IN
        y_centroid = np.sum(y_IN)/Npts_IN
        x_IN = x_IN - x_centroid; X = X - x_centroid
        y_IN = y_IN - y_centroid; Y = Y - y_centroid

        ##### Scaling - dividing by twice the distance of farthest point from origin
        d_IN = np.sqrt(x_IN**2 + y_IN**2)
        d_char = 2.*np.max(d_IN)
        x_IN = x_IN/d_char; X = X/d_char
        y_IN = y_IN/d_char; Y = Y/d_char
    else:
        d_char = 1

    ##### Determine which polynomial to fit based upon the number of input points
    if Npts_IN == 7:
        # f7(x,y) = a(1) + b(Y) + c(X) + d(Y^2) + e(X^2) + f(YX) + g(Y^3)
        v = np.array([np.ones(Npts_IN),y_IN,x_IN,y_IN**2,x_IN**2,y_IN*x_IN,y_IN**3])
    elif Npts_IN == 8:
        # f7(x,y) = a(1) + b(X) + c(Y) + d(X^2) + e(Y^2) + f(XY) + g(X^3) + h(Y^3)
        v = np.array([np.ones(Npts_IN),y_IN,x_IN,y_IN**2,x_IN**2,x_IN*y_IN,y_IN**3,x_IN**3])
    elif Npts_IN == 9:
        # f7(x,y) = a(1) + b(X) + c(Y) + d(X^2) + e(Y^2) + f(XY) + g(X^3) + h(Y^3) + i(Y^2*X)
        v = np.array([np.ones(Npts_IN),y_IN,x_IN,y_IN**2,x_IN**2,x_IN*y_IN,y_IN**3,x_IN**3,y_IN*x_IN*y_IN])
    elif Npts_IN == 10:
        # f7(x,y) = a(1) + b(X) + c(Y) + d(X^2) + e(Y^2) + f(XY) + g(X^3) + h(Y^3) + i(Y^2*X)
        v = np.array([np.ones(Npts_IN),y_IN,x_IN,y_IN**2,x_IN**2,x_IN*y_IN,y_IN**3,x_IN**3,y_IN*x_IN*y_IN,x_IN*x_IN*y_IN])
    elif Npts_IN == 11:
        # f7(x,y) = a(1) + b(X) + c(Y) + d(X^2) + e(Y^2) + f(XY) + g(X^3) + h(Y^3) + i(Y^2*X)
        v = np.array([np.ones(Npts_IN),y_IN,x_IN,y_IN**2,x_IN**2,x_IN*y_IN,y_IN**3,x_IN**3,y_IN*x_IN*y_IN,x_IN*x_IN*y_IN,y_IN**4])
    elif Npts_IN == 6:
        # f6(x,y) = a(1) + b(Y) + c(X) + d(Y^2) + e(X^2) + f(YX)
        v = np.array([np.ones(Npts_IN),y_IN,x_IN,y_IN**2,x_IN**2,y_IN*x_IN])
    elif Npts_IN == 5:
        # f5(x,y) = a(1) + b(Y) + c(X) + d(Y^2) + e(X^2)
        v = np.array([np.ones(Npts_IN),y_IN,x_IN,y_IN**2,x_IN**2])
    elif Npts_IN == 4:
        # f4(x,y) = a(1) + b(Y) + c(X) + d(Y^2)
        v = np.array([np.ones(Npts_IN),y_IN,x_IN,y_IN**2])
    elif Npts_IN == 3:
        # f3(x,y) = a(1) + b(Y) + c(X)
        v = np.array([np.ones(Npts_IN),y_IN,x_IN])

    else:
        return 0,0,0
    
    coeff_der0,_,_,_ = np.linalg.lstsq(v.T, phi_IN)
    coeff_der1y    = np.zeros(coeff_der0.shape)
    coeff_der2y    = np.zeros(coeff_der0.shape)

    if Npts_IN == 7:
        # f7(x,y)  =  a(1)  + b(Y) + c(X) + d(Y^2) + e(X^2) + f(YX) + g(Y^3)
        #            [0]     [1]    [2]    [3]      [4]      [5]     [6]

        #            [1]     [3]    [5]     [6]
        # df7/dy   =  b(1) + 2d(Y) + f(X) + 3g(Y^2)
        #              [0]     [1]    [2]     [3]
        coeff_der1y[0] = np.array(coeff_der0)[1]
        coeff_der1y[1] = 2 * np.array(coeff_der0)[3]
        coeff_der1y[2] = np.array(coeff_der0)[5]
        coeff_der1y[3] = 3 * np.array(coeff_der0)[6]
        
        #            [3]     [6]
        # d2f7/dy2 = 2d(1) + 6g(Y)
        #              [0]     [1]
        coeff_der2y[0] = 2 * np.array(coeff_der0)[3]
        coeff_der2y[1] = 6 * np.array(coeff_der0)[6]

        vv = np.array([np.ones(Npts_OUT),Y,X,Y**2,X**2,Y*X,Y**3])

    elif Npts_IN == 8:
        # f7(x,y)  =  a(1)  + b(X) + c(Y) + d(X^2) + e(Y^2) + f(XY) + g(Y^3) + g(X^3)
        #            [0]     [1]    [2]    [3]      [4]      [5]      [6]       [7]

        #            [1]     [3]    [5]     [6]
        # df7/dy   =  b(1) + 2d(Y) + f(X) + 3g(Y^2)
        #              [0]     [1]    [2]     [3]
        coeff_der1y[0] = np.array(coeff_der0)[1]
        coeff_der1y[1] = 2 * np.array(coeff_der0)[3]
        coeff_der1y[2] = np.array(coeff_der0)[5]
        coeff_der1y[3] = 3 * np.array(coeff_der0)[6]
        
        #            [3]     [6]
        # d2f7/dy2 = 2d(1) + 6g(Y)
        #              [0]     [1]
        coeff_der2y[0] = 2 * np.array(coeff_der0)[3]
        coeff_der2y[1] = 6 * np.array(coeff_der0)[6]

        vv = np.array([np.ones(Npts_OUT),Y,X,Y**2,X**2,Y*X,Y**3,X**3])

    elif Npts_IN == 9:
        # f7(x,y)  =  a(1)  + b(X) + c(Y) + d(X^2) + e(Y^2) + f(XY) + g(Y^3) + g(X^3) + i(Y^2X) 
        #            [0]     [1]    [2]    [3]      [4]      [5]      [6]       [7]     [8]

        #            [1]     [3]    [5]     [6]
        # df7/dx   =  b(1) + 2d(Y) + f(X) + 3g(Y^2) + 2i(XY)
        #              [0]     [1]    [2]     [3]
        coeff_der1y[0] = np.array(coeff_der0)[1]
        coeff_der1y[1] = 2 * np.array(coeff_der0)[3]
        coeff_der1y[2] = np.array(coeff_der0)[5]
        coeff_der1y[3] = 3 * np.array(coeff_der0)[6]
        coeff_der1y[4] = 2 * np.array(coeff_der0)[8]
        #            [3]     [6]
        # d2f7/dx2 = 2d(1) + 6g(Y) + 2i(X)
        #              [0]     [1]
        coeff_der2y[0] = 2 * np.array(coeff_der0)[3]
        coeff_der2y[1] = 6 * np.array(coeff_der0)[6]
        coeff_der2y[2] = 2 * np.array(coeff_der0)[8]

        vv = np.array([np.ones(Npts_OUT),Y,X,Y**2,X**2,Y*X,Y**3 ,X**3,Y*Y*X])
    elif Npts_IN == 10:
        # f7(x,y)  =  a(1)  + b(X) + c(Y) + d(X^2) + e(Y^2) + f(XY) + g(Y^3) + g(X^3) + i(Y^2X) 
        #            [0]     [1]    [2]    [3]      [4]      [5]      [6]       [7]     [8]

        #            [1]     [3]    [5]     [6]
        # df7/dx   =  b(1) + 2d(Y) + f(X) + 3g(Y^2) + 2i(XY)
        #              [0]     [1]    [2]     [3]
        coeff_der1y[0] = np.array(coeff_der0)[1]
        coeff_der1y[1] = 2 * np.array(coeff_der0)[3]
        coeff_der1y[2] = np.array(coeff_der0)[5]
        coeff_der1y[3] = 3 * np.array(coeff_der0)[6]
        coeff_der1y[4] = 2 * np.array(coeff_der0)[8]
        coeff_der1y[5] = np.array(coeff_der0)[8]
        #            [3]     [6]
        # d2f7/dx2 = 2d(1) + 6g(Y) + 2i(X)
        #              [0]     [1]
        coeff_der2y[0] = 2 * np.array(coeff_der0)[3]
        coeff_der2y[1] = 6 * np.array(coeff_der0)[6]
        coeff_der2y[2] = 2 * np.array(coeff_der0)[8]

        vv = np.array([np.ones(Npts_OUT),Y,X,Y**2,X**2,Y*X,Y**3,X**3,Y*Y*X,X*X*Y])
    elif Npts_IN == 11:
        # f7(x,y)  =  a(1)  + b(X) + c(Y) + d(X^2) + e(Y^2) + f(XY) + g(X^3) + h(Y^3) + i(X^2Y) +  j(Y^2X) + k(Y^4)
        #            [0]     [1]    [2]    [3]      [4]      [5]      [6]       [7]     [8]    +    [9]    +    [10]

        #            [1]     [3]    [5]     [6]
        # df7/dx   =  b(1) + 2d(X) + f(Y) + 3g(X^2) + 2i(XY)
        #              [0]     [1]    [2]     [3]
        coeff_der1y[0] = np.array(coeff_der0)[1]
        coeff_der1y[1] = 2 * np.array(coeff_der0)[3]
        coeff_der1y[2] = np.array(coeff_der0)[5]
        coeff_der1y[3] = 3 * np.array(coeff_der0)[6]
        coeff_der1y[4] = 2 * np.array(coeff_der0)[8]
        coeff_der1y[5] = np.array(coeff_der0)[9]
        coeff_der1y[6] = 4 * np.array(coeff_der0)[10]
        #            [3]     [6]
        # d2f7/dx2 = 2d(1) + 6g(X) + 2i(Y)
        #              [0]     [1]
        coeff_der2y[0] = 2 * np.array(coeff_der0)[3]
        coeff_der2y[1] = 6 * np.array(coeff_der0)[6]
        coeff_der2y[2] = 2 * np.array(coeff_der0)[8]
        coeff_der2y[3] = 12 * np.array(coeff_der0)[10]

        vv = np.array([np.ones(Npts_OUT),Y,X,Y**2,X**2,X*Y,Y**3,X**3,Y*X*Y,X*Y*X,Y**4])
    elif Npts_IN == 6:
        # f6(x,y)  =  a(1)  + b(Y) + c(X) + d(Y^2) + e(X^2) + f(YX)
        #            [0]     [1]    [2]    [3]      [4]      [5]

        #            [1]     [3]    [5]
        # df6/dy   =  b(1) + 2d(Y) + f(X)
        #              [0]     [1]    [2]
        coeff_der1y[0] = np.array(coeff_der0)[1]
        coeff_der1y[1] = 2 * np.array(coeff_der0)[3]
        coeff_der1y[2] = np.array(coeff_der0)[5]
        
        #            [3]
        # d2f6/dy2 = 2d(1)
        #              [0]
        coeff_der2y[0] = 2 * np.array(coeff_der0)[3]

        vv = np.array([np.ones(Npts_OUT),Y,X,Y**2,X**2,Y*X])

        
    elif Npts_IN == 5:
        # f5(x,y)  =  a(1)  + b(Y) + c(X) + d(Y^2) + e(X^2)
        #            [0]     [1]    [2]    [3]      [4]

        #            [1]     [3]
        # df5/dy   =  b(1) + 2d(Y)
        #              [0]     [1]
        coeff_der1y[0] = np.array(coeff_der0)[1]
        coeff_der1y[1] = 2 * np.array(coeff_der0)[3]
        #coeff_der1y[2] = np.array(coeff_der0)[4]

        #            [3]
        # d2f5/dy2 = 2d(1)
        #              [0]
        coeff_der2y[0] = 2 * np.array(coeff_der0)[3]

        vv = np.array([np.ones(Npts_OUT),Y,X,Y**2,X**2])
        
    elif Npts_IN == 4:
        # f4(x,y)  =  a(1)  + b(Y) + c(X) + d(Y^2)
        #            [0]     [1]    [2]    [3]
        
        #            [1]     [3]
        # df4/dy   =  b(1) + 2d(Y)
        #              [0]     [1]
        coeff_der1y[0] = np.array(coeff_der0)[1]
        coeff_der1y[1] = 2 * np.array(coeff_der0)[3]
        
        #            [3]
        # d2f4/dy2 = 2d(1)
        #              [0]
        coeff_der2y[0] = 2 * np.array(coeff_der0)[3]

        vv = np.array([np.ones(Npts_OUT),Y,X,Y**2])
        
    elif Npts_IN == 3:
        # f3(x,y)  = a(1) + b(Y) + c(X)
        #           [0]    [1]    [2]
        
        #           [1]
        # df3/dy   = b(1)
        #             [0]
        coeff_der1y[0] = np.array(coeff_der0)[1]

        # d2f3/dy2 = 0(1)
        #             [0]
        coeff_der2y[0] = 0

        vv = np.array([np.ones(Npts_OUT),Y,X])
  
    phi      = coeff_der0.dot(vv)
    dphi_dy  = coeff_der1y.dot(vv)/d_char
    d2phi_dy2  = coeff_der2y.dot(vv)/(d_char**2)

    return phi,dphi_dy,d2phi_dy2