import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pdb import set_trace as keyboard

figure_folder = "../report/figures/"

def plot_data(x,y,phi,b):
    #x = ... # x coordinates of data
    #y = ... # y coordinates of data
    #phi = .. # values of data
    # define regular grid spatially covering input data
    n = 100
    xg = np.linspace(x.min(),x.max(),n)
    yg = np.linspace(y.min(),y.max(),n)
    X,Y = np.meshgrid(xg,yg)
    # interpolate Z values on defined grid
    Z = griddata(np.vstack((x.flatten(),y.flatten())).T, np.vstack(phi.flatten()),(X,Y),method="cubic").reshape(X.shape)
    circle_1 = plt.Circle((1.3,0.5),0.1,color='k')
    circle_2 = plt.Circle((1.7,0.3),0.1,color='k')
    circle_3 = plt.Circle((1.7,0.7),0.1,color='k')
##    keyboard()
    # mask nan values, so they will not appear on plot
    Zm = np.ma.masked_where(np.isnan(Z),Z)
    # plot
    fig = plt.figure()
    fig.add_subplot(111, aspect="equal")
    plt.pcolormesh(X,Y,Zm,shading='gourand')
    plt.gcf().gca().add_artist(circle_1)
    plt.gcf().gca().add_artist(circle_2)
    plt.gcf().gca().add_artist(circle_3)
    plt.colorbar()
    plt.tight_layout()
    figure_name = figure_folder + b
    print "Saving figure: "+ figure_name
    plt.savefig(figure_name)
    plt.close()