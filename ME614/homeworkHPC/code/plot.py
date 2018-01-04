import os
import sys
import numpy as np   # library to handle arrays like Matlab
import scipy.sparse as scysparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pickle

#Part a
figure_name = "Part_a.pdf"
figure_folder = "../report/"
figure_file_path = figure_folder + figure_name

figwidth       = 18
figheight      = 16
lineWidth      = 3
textFontSize   = 28
gcafontSize    = 30


fig = plt.figure(1, figsize=(figwidth,figheight))

ax_1 = fig.add_subplot(111)

f = open("Truncation_Error_Np=1.p","rb")
error_TR1 = pickle.load(f)
f.close()
f = open("Truncation_Error_Np=2.p","rb")
error_TR2 = pickle.load(f)
f.close()
f = open("Truncation_Error_Np=4.p","rb")
error_TR3 = pickle.load(f)
f.close()
f = open("Truncation_Error_Np=8.p","rb")
error_TR4 = pickle.load(f)
f.close()
f = open("Truncation_Error_Np=16.p","rb")
error_TR5 = pickle.load(f)
f.close()


f1 = open("Grid_Spacing.p","rb")
dx = pickle.load(f1)
f1.close()
dx_inv = 1/dx

ax = ax_1
plt.axes(ax)
ax.loglog(dx_inv,error_TR1,'-r',linewidth=lineWidth)
ax.loglog(dx_inv,error_TR2,'-b',linewidth=lineWidth)
ax.loglog(dx_inv,error_TR3,'-g',linewidth=lineWidth)
ax.loglog(dx_inv,error_TR4,'-k',linewidth=lineWidth)
ax.loglog(dx_inv,error_TR5,'-m',linewidth=lineWidth)
ax.loglog(dx_inv,dx**3, '--')
ax.loglog(dx_inv,dx**4, '--')
ax.loglog(dx_inv,dx**5, '--')
plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
ax.grid('on',which='both')

ax.set_xlabel(r"$dx^{-1}$",fontsize=textFontSize)
ax.set_ylabel(r"$epsilon_{TR}$",fontsize=textFontSize,rotation=90)
plt.title("Truncation Error vs Inverse Grid Spacing",fontsize=textFontSize)
plt.legend(["N=1","N=2","N=4","N=8", "N=16", "3rd Order", "4th Order", "5th Order",],loc='best')
print "Saving Figure:" + figure_file_path
plt.savefig(figure_file_path)


#Part b, Strong scaling plots

comp_time = np.zeros(5,dtype=np.float128)
n_processors = np.array([1, 2, 4, 8, 16])
efficiency = np.zeros(5,dtype=np.float128)
for i,ct in enumerate(comp_time):

	f = open("Calculation_Time_N=10^8+1_Np=%d.p" %(n_processors[i]), "rb")
	comp_time[i] = pickle.load(f)
	f.close()
	efficiency[i] = comp_time[0]*100/(n_processors[i]*comp_time[i])
	print efficiency[i]

figure_name1 = "Part_b_Strong_Scaling.pdf"
figure_folder = "../report/"
figure_file_path = figure_folder + figure_name1

figwidth       = 18
figheight      = 16
lineWidth      = 3
textFontSize   = 28
gcafontSize    = 30

fig = plt.figure(2, figsize=(figwidth,figheight))
ax_2 = fig.add_subplot(111)
ax = ax_2
plt.axes(ax)
ax.plot(n_processors,efficiency,'-k',linewidth=lineWidth)
ax.plot(n_processors,50*np.ones(5),'--r',linewidth=lineWidth)
ax.plot(n_processors,75*np.ones(5),'--g',linewidth=lineWidth)
ax.plot(n_processors,100*np.ones(5),'--b',linewidth=lineWidth)
plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
ax.grid('on',which='both')

ax.set_xlabel("Number of Processors",fontsize=textFontSize)
ax.set_ylabel("Strong Scaling Efficiency (%)",fontsize=textFontSize,rotation=90)
plt.title("Strong Scaling Efficiency vs No. of Processors",fontsize=textFontSize)
plt.legend(["Strong scaling efficiency","50%","75%","100%"])
print "Saving Figure:" + figure_file_path
plt.savefig(figure_file_path)


#Part b Weak Scaling

comp_time = np.zeros(5)
n_processors = np.array([1, 2, 4, 8, 16])
efficiency = np.zeros(5)
for i,ct in enumerate(comp_time):

	f2 = open("Calculation_Time_Partb_Np=%d.p" %(n_processors[i]), "rb")
	comp_time[i] = pickle.load(f2)
	f2.close()
	efficiency[i] = comp_time[0]*100/comp_time[i]
	print efficiency[i]

figure_name = "Part_b_Weak_Scaling.pdf"
figure_folder = "../report/"
figure_file_path = figure_folder + figure_name

figwidth       = 18
figheight      = 16
lineWidth      = 3
textFontSize   = 28
gcafontSize    = 30

fig = plt.figure(3, figsize=(figwidth,figheight))
ax_3 = fig.add_subplot(111)
ax = ax_3
plt.axes(ax)
ax.plot(n_processors,efficiency,'-k',linewidth=lineWidth)
ax.plot(n_processors,50*np.ones(5),'--r',linewidth=lineWidth)
ax.plot(n_processors,75*np.ones(5),'--g',linewidth=lineWidth)
ax.plot(n_processors,100*np.ones(5),'--b',linewidth=lineWidth)
plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
ax.grid('on',which='both')

ax.set_xlabel("Number of Processors",fontsize=textFontSize)
ax.set_ylabel("Weak Scaling Efficieny (%)",fontsize=textFontSize,rotation=90)
plt.title("Weak Scaling Efficiency vs No. of Processors",fontsize=textFontSize)
plt.legend(["Weak scaling efficiency","50%","75%","100%"])
print "Saving Figure:" + figure_file_path
plt.savefig(figure_file_path)

#Part c

figure_name = "Part_c.pdf"
figure_folder = "../report/"
figure_file_path = figure_folder + figure_name

figwidth       = 18
figheight      = 16
lineWidth      = 3
textFontSize   = 28
gcafontSize    = 30


fig = plt.figure(4, figsize=(figwidth,figheight))

ax_4 = fig.add_subplot(111)

index = np.arange(1,17,1)
Trunc_error = np.zeros(index.size, dtype=np.float128)
for i,p in enumerate(index):

	f = open("Truncation_Error_partc_Np=%d.p" % (p),"rb")
	Trunc_error[i] = pickle.load(f)
	f.close()

ax = ax_4
plt.axes(ax)
ax.semilogy(index,Trunc_error,'-k',linewidth=lineWidth)
avg = np.sum(Trunc_error,dtype=np.float128)/(index.size)*np.ones(index.size)
ax.semilogy(index,avg,'-r',linewidth=lineWidth)
plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
ax.grid('on',which='both')

ax.set_xlabel("Number of Processors",fontsize=textFontSize)
ax.set_ylabel(r"$epsilon_{TR}$",fontsize=textFontSize,rotation=90)
plt.title("Truncation Error vs Number of Processors",fontsize=textFontSize)
plt.legend(["Truncation Error","Average"])
print "Saving Figure:" + figure_file_path
plt.savefig(figure_file_path)
