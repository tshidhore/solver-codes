import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

figure_folder = "../report/"

def plot_data(x,y,phi,b):
    #x = ... # x coordinates of data
    #y = ... # y coordinates of data
    #phi = .. # values of data
    # define regular grid spatially covering input data
    n = 50
    xg = np.linspace(x.min(),x.max(),n)
    yg = np.linspace(y.min(),y.max(),n)
    X,Y = np.meshgrid(xg,yg)
    # interpolate Z values on defined grid
    Z = griddata(np.vstack((x.flatten(),y.flatten())).T, np.vstack(phi.flatten()),(X,Y),method="cubic").reshape(X.shape)
    # mask nan values, so they will not appear on plot
    Zm = np.ma.masked_where(np.isnan(Z),Z)
    # plot
    fig = plt.figure()
    fig.add_subplot(111, aspect="equal")
    plt.pcolormesh(X,Y,Zm,shading='gourand')
    plt.colorbar()
    plt.tight_layout()
    figure_name = figure_folder + b
    print "Saving figure: "+ figure_name
    plt.savefig(figure_name)
    plt.close()