import os
import sys
import numpy as np
import scipy as sp 
import scipy.sparse as scysparse
import scipy.sparse.linalg as splinalg
import pylab as plt
from pdb import set_trace
import umesh_reader
import plot_data
from pdb import set_trace as keyboard
import time

plt.close('all')

#keyboard()
p2a = True
p2b = True
p2ex = True
p3 = True


#####################################################################################################################
if p2a:
    
#    Part a
    icemcfd_project_folder = './'
#    filename = '/mesh_set_1/heated_rod_ncv=103.msh'
    filename = 'mesh_set/Mesh3.msh'
    figure_folder = "../report/"

    mshfile_fullpath = icemcfd_project_folder + filename

    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)    
    
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    cold_bc = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'COLD'
    hot_bc = np.where(partofa1=='HOT')   # Vectorized approach to find face belonging to part 'HOT'
    
    #Temperature initialized at nodes and CVs
    Tn = 300*np.ones(nno)    
    
    # Initializing values for hot and cold boundary nodes
    Tn[np.unique(noofa[cold_bc])] = 300 
    Tn[np.unique(noofa[hot_bc])] = 500
    
    plot_data.plot_data(xy_no[:,0],xy_no[:,1],Tn,"Initial Temperature Field.pdf")
    # Initializing CV temperatures at 300
    Tcv = 300*np.ones(ncv)
    
    n = 1 # Counter for no. of iterations
    Nmax = 3000 #max no. of iterations
    step_print = 500 # Interation step when contour is extracted
    
    tol = 0.01 #Convergence criterion for RMS
    RMS = 10 #Defn of RMS value

    while RMS>=tol and n <= Nmax:
        
        T_old = list(Tn)
        # Looping over each CV
        for i in np.arange(ncv):
            Tcv[i] = np.average(Tn[np.unique(noofa[faocv[i]])]) # Averaging surrounding nodal termperatures and storing them in Tcv. Note that np.unique ensures that each node is accounted for only once
        for j in np.arange(nno):
            flag = 0 # flag to check if node is boundary node 
            fa1 = partofa1[faono[j]] # corresponding array of parts to which the faces containing the node belong to
            for k,pn in enumerate(fa1): # Checking if any of the part names correspond to the boundary
                if pn == 'COLD':
                    flag = 1
                if pn == 'HOT':
                    flag = 1
            if flag == 0: # faces which do not belong to the boundary
                Tn[j] = np.average(Tcv[np.unique(cvofa[faono[j]])]) # Averaging surrounding CV termperatures and storing them in Tn. Note that np.unique ensures that each CV is accounted for only once
        print "Iteration: %d" %(n)
        RMS = np.sqrt(sum(Tn - T_old)**2/nno)
        print "RMS Error: %2.7f" %(RMS)
        if n%step_print == 0 and n!=Nmax:
            plot_data.plot_data(xy_no[:,0],xy_no[:,1],Tn,"Temperature Contour at n=%d.pdf" %(n)) 
        n += 1 # Coded for 10000 iterations   
    plot_data.plot_data(xy_no[:,0],xy_no[:,1],Tn,"Final Temperature Contour, n=%d.pdf" %(n))
    
if p2b:
    
    # Part B
    icemcfd_project_folder = './'
    #filename = 'heated_rod_ncv=103.msh'
    filename = 'mesh_set/Mesh3.msh'
    figure_folder = "../report/"
    mshfile_fullpath = icemcfd_project_folder + filename

    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)    
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    cold_bc = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'COLD'
    hot_bc = np.where(partofa1=='HOT')   # Vectorized approach to find face belonging to part 'HOT'
    solid = np.where(partofa1=='SOLID')
    
    # Node-to-CV conversion sparse matrix
    An2cv = scysparse.csr_matrix((ncv,nno),dtype="float64")
    
    # CV-to-node conversion matrix for internal nodes. Note that boundary nodes will be treated within the matrix itself.
    Acv2n_int = scysparse.csr_matrix((nno-np.unique(noofa[cold_bc]).size-np.unique(noofa[hot_bc]).size,ncv),dtype="float64")  # Only takes interior points
    
    for i in np.arange(ncv):
        
        nnn = np.unique(noofa[faocv[i]]).size # Gives number of neighbouring nodes, for calculating weight for each position = 1/no. of surrounding nodes
        An2cv[i,np.unique(noofa[faocv[i]])] = 1./nnn
    # This assumes that boundary nodes are numbered the towards the very end. If not, then God help you!!
    for i in np.arange(nno):
        nncv = np.unique(cvofa[faono[i]]).size  # No. of neighbouring CVs of the ndoe
        flag = 0 # flag to check if node is boundary node 
        fa1 = partofa1[faono[i]] # corresponding array of parts to which the faces containing the node belong to
        for k,pn in enumerate(fa1): # Checking if any of the part names correspond to the boundary
            if pn == 'COLD':
                flag = 1
            if pn == 'HOT':
                flag = 1
        if flag == 0: # faces which do not belong to the boundary
            Acv2n_int[i,np.unique(cvofa[faono[i]])] = 1./nncv # Averaging surrounding CV termperatures and storing them in Tn. Note that np.unique ensures that each CV is accounted for only once         
    
    plt.spy(An2cv)
    name = "Spy plot of An2cv.pdf"
    figure_name = figure_folder + name
    print "Saving figure: "+ figure_name
    plt.savefig(figure_name)
    plt.close()
    
    plt.spy(Acv2n_int)
    name = "Spy plot of Acv2n_int.pdf"
    figure_name = figure_folder + name
    print "Saving figure: "+ figure_name
    plt.savefig(figure_name)
    plt.close()
     
if p2ex:
    
    #Mesh1
    print "Mesh A"
    icemcfd_project_folder = './'
#    filename = '/mesh_set_1/heated_rod_ncv=103.msh'
    filename = 'mesh_set/Mesh1.msh'
    figure_folder = "../report/"

    mshfile_fullpath = icemcfd_project_folder + filename

    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)
    start_time2 = time.time()
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    cold_bc = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'COLD'
    hot_bc = np.where(partofa1=='HOT')   # Vectorized approach to find face belonging to part 'HOT'
    solid = np.where(partofa1=='SOLID')
    
    #Temperature initialized at nodes and CVs
    Tn = np.zeros(nno)    
    
    # Initializing values for hot and cold boundary nodes
    Tn[np.unique(noofa[cold_bc])] = 300 
    Tn[np.unique(noofa[hot_bc])] = 500
    
    # Initializing CV temperatures
    Tcv = np.zeros(ncv)
    
    # Node-to-CV conversion sparse matrix
    An2cv = scysparse.csr_matrix((ncv,nno),dtype="float64")
    
    # CV-to-node conversion matrix for internal nodes. Note that boundary nodes will be treated within the matrix itself.
    Acv2n_int = scysparse.csr_matrix((nno-np.unique(noofa[cold_bc]).size-np.unique(noofa[hot_bc]).size,ncv),dtype="float64")  # Only takes interior points
    
    for i in np.arange(ncv):
        
        nnn = np.unique(noofa[faocv[i]]).size # Gives number of neighbouring nodes, for calculating weight for each position = 1/no. of surrounding nodes
        An2cv[i,np.unique(noofa[faocv[i]])] = 1./nnn
    # This assumes that boundary nodes are numbered the towards the very end. If not, then God help you!!
    for i in np.arange(nno):
        nncv = np.unique(cvofa[faono[i]]).size  # No. of neighbouring CVs of the ndoe
        flag = 0 # flag to check if node is boundary node 
        fa1 = partofa1[faono[i]] # corresponding array of parts to which the faces containing the node belong to
        for k,pn in enumerate(fa1): # Checking if any of the part names correspond to the boundary
            if pn == 'COLD':
                flag = 1
            if pn == 'HOT':
                flag = 1
        if flag == 0: # faces which do not belong to the boundary
            Acv2n_int[i,np.unique(cvofa[faono[i]])] = 1./nncv # Averaging surrounding CV termperatures and storing them in Tn. Note that np.unique ensures that each CV is accounted for only once        
    n = 1 # Counter for no. of iterations
     
    while n <= 500:
        
        # Updating CV centre values 
        Tcv = An2cv.dot(Tn)
        
        #Updating node values based on CV centres
        Tn_int = Acv2n_int.dot(Tcv)        
        
        #Assuming that All boundary nodes are loacted towards the very end of Tn
        Tn[0:Tn_int.size] = Tn_int        
        
#        print "Iteration: %d" %(n)
        n += 1 # Coded for 10000 iterations
    
    stop_time2 = time.time() - start_time2    
    print "Time required to execute matrix solve for Mesh A: %2.10f" %(stop_time2)


    start_time1 = time.time()  # Time Starts
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    cold_bc = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'COLD'
    hot_bc = np.where(partofa1=='HOT')   # Vectorized approach to find face belonging to part 'HOT'
    
    #Temperature initialized at nodes and CVs
    Tn = np.zeros(nno)    
    
    # Initializing values for hot and cold boundary nodes
    Tn[np.unique(noofa[cold_bc])] = 300 
    Tn[np.unique(noofa[hot_bc])] = 500

    # Initializing CV temperatures
    Tcv = np.zeros(ncv)
    
    n = 1 # Counter for no. of iterations
    
    while n <= 500:
        # Looping over each CV
        for i in np.arange(ncv):
            Tcv[i] = np.average(Tn[np.unique(noofa[faocv[i]])]) # Averaging surrounding nodal termperatures and storing them in Tcv. Note that np.unique ensures that each node is accounted for only once
        for j in np.arange(nno):
            flag = 0 # flag to check if node is boundary node 
            fa1 = partofa1[faono[j]] # corresponding array of parts to which the faces containing the node belong to
            for k,pn in enumerate(fa1): # Checking if any of the part names correspond to the boundary
                if pn == 'COLD':
                    flag = 1
                if pn == 'HOT':
                    flag = 1
            if flag == 0: # faces which do not belong to the boundary
                Tn[j] = np.average(Tcv[np.unique(cvofa[faono[j]])]) # Averaging surrounding CV termperatures and storing them in Tn. Note that np.unique ensures that each CV is accounted for only once
#        print "Iteration: %d" %(n)
        n += 1 # Coded for 10000 iterations
    stop_time1 = time.time() - start_time1    
    print "Time required to execute for looping for Mesh A: %2.10f" %(stop_time1)

    #Mesh2
    print "Mesh B"
    icemcfd_project_folder = './'
#    filename = '/mesh_set_1/heated_rod_ncv=103.msh'
    filename = 'mesh_set/Mesh2.msh'
    figure_folder = "../report/"

    mshfile_fullpath = icemcfd_project_folder + filename

    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)
    start_time2 = time.time()
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    cold_bc = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'COLD'
    hot_bc = np.where(partofa1=='HOT')   # Vectorized approach to find face belonging to part 'HOT'
    solid = np.where(partofa1=='SOLID')
    
    #Temperature initialized at nodes and CVs
    Tn = np.zeros(nno)    
    
    # Initializing values for hot and cold boundary nodes
    Tn[np.unique(noofa[cold_bc])] = 300 
    Tn[np.unique(noofa[hot_bc])] = 500
    
    # Initializing CV temperatures
    Tcv = np.zeros(ncv)
    
    # Node-to-CV conversion sparse matrix
    An2cv = scysparse.csr_matrix((ncv,nno),dtype="float64")
    
    # CV-to-node conversion matrix for internal nodes. Note that boundary nodes will be treated within the matrix itself.
    Acv2n_int = scysparse.csr_matrix((nno-np.unique(noofa[cold_bc]).size-np.unique(noofa[hot_bc]).size,ncv),dtype="float64")  # Only takes interior points
    
    for i in np.arange(ncv):
        
        nnn = np.unique(noofa[faocv[i]]).size # Gives number of neighbouring nodes, for calculating weight for each position = 1/no. of surrounding nodes
        An2cv[i,np.unique(noofa[faocv[i]])] = 1./nnn
    # This assumes that boundary nodes are numbered the towards the very end. If not, then God help you!!
    for i in np.arange(nno):
        nncv = np.unique(cvofa[faono[i]]).size  # No. of neighbouring CVs of the ndoe
        flag = 0 # flag to check if node is boundary node 
        fa1 = partofa1[faono[i]] # corresponding array of parts to which the faces containing the node belong to
        for k,pn in enumerate(fa1): # Checking if any of the part names correspond to the boundary
            if pn == 'COLD':
                flag = 1
            if pn == 'HOT':
                flag = 1
        if flag == 0: # faces which do not belong to the boundary
            Acv2n_int[i,np.unique(cvofa[faono[i]])] = 1./nncv # Averaging surrounding CV termperatures and storing them in Tn. Note that np.unique ensures that each CV is accounted for only once        
    n = 1 # Counter for no. of iterations
     
    while n <= 500:
        
        # Updating CV centre values 
        Tcv = An2cv.dot(Tn)
        
        #Updating node values based on CV centres
        Tn_int = Acv2n_int.dot(Tcv)        
        
        #Assuming that All boundary nodes are loacted towards the very end of Tn
        Tn[0:Tn_int.size] = Tn_int        
        
#        print "Iteration: %d" %(n)
        n += 1 # Coded for 10000 iterations
    
    stop_time2 = time.time() - start_time2    
    print "Time required to execute matrix solve for Mesh B: %2.10f" %(stop_time2)


    start_time1 = time.time()  # Time Starts
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    cold_bc = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'COLD'
    hot_bc = np.where(partofa1=='HOT')   # Vectorized approach to find face belonging to part 'HOT'
    
    #Temperature initialized at nodes and CVs
    Tn = np.zeros(nno)    
    
    # Initializing values for hot and cold boundary nodes
    Tn[np.unique(noofa[cold_bc])] = 300 
    Tn[np.unique(noofa[hot_bc])] = 500
    
    # Initializing CV temperatures
    Tcv = np.zeros(ncv)
    
    n = 1 # Counter for no. of iterations
    
    while n <= 500:
        # Looping over each CV
        for i in np.arange(ncv):
            Tcv[i] = np.average(Tn[np.unique(noofa[faocv[i]])]) # Averaging surrounding nodal termperatures and storing them in Tcv. Note that np.unique ensures that each node is accounted for only once
        for j in np.arange(nno):
            flag = 0 # flag to check if node is boundary node 
            fa1 = partofa1[faono[j]] # corresponding array of parts to which the faces containing the node belong to
            for k,pn in enumerate(fa1): # Checking if any of the part names correspond to the boundary
                if pn == 'COLD':
                    flag = 1
                if pn == 'HOT':
                    flag = 1
            if flag == 0: # faces which do not belong to the boundary
                Tn[j] = np.average(Tcv[np.unique(cvofa[faono[j]])]) # Averaging surrounding CV termperatures and storing them in Tn. Note that np.unique ensures that each CV is accounted for only once
#        print "Iteration: %d" %(n)
        n += 1 # Coded for 10000 iterations
    stop_time1 = time.time() - start_time1    
    print "Time required to execute for looping for Mesh B: %2.10f" %(stop_time1)

    #Mesh3
    print "Mesh C"
    icemcfd_project_folder = './'
#    filename = '/mesh_set_1/heated_rod_ncv=103.msh'
    filename = 'mesh_set/Mesh3.msh'
    figure_folder = "../report/"

    mshfile_fullpath = icemcfd_project_folder + filename

    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)
    start_time2 = time.time()
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    cold_bc = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'COLD'
    hot_bc = np.where(partofa1=='HOT')   # Vectorized approach to find face belonging to part 'HOT'
    solid = np.where(partofa1=='SOLID')
    
    #Temperature initialized at nodes and CVs
    Tn = np.zeros(nno)    
    
    # Initializing values for hot and cold boundary nodes
    Tn[np.unique(noofa[cold_bc])] = 300 
    Tn[np.unique(noofa[hot_bc])] = 500

    # Initializing CV temperatures
    Tcv = np.zeros(ncv)
    
    # Node-to-CV conversion sparse matrix
    An2cv = scysparse.csr_matrix((ncv,nno),dtype="float64")
    
    # CV-to-node conversion matrix for internal nodes. Note that boundary nodes will be treated within the matrix itself.
    Acv2n_int = scysparse.csr_matrix((nno-np.unique(noofa[cold_bc]).size-np.unique(noofa[hot_bc]).size,ncv),dtype="float64")  # Only takes interior points
    
    for i in np.arange(ncv):
        
        nnn = np.unique(noofa[faocv[i]]).size # Gives number of neighbouring nodes, for calculating weight for each position = 1/no. of surrounding nodes
        An2cv[i,np.unique(noofa[faocv[i]])] = 1./nnn
    # This assumes that boundary nodes are numbered the towards the very end. If not, then God help you!!
    for i in np.arange(nno):
        nncv = np.unique(cvofa[faono[i]]).size  # No. of neighbouring CVs of the ndoe
        flag = 0 # flag to check if node is boundary node 
        fa1 = partofa1[faono[i]] # corresponding array of parts to which the faces containing the node belong to
        for k,pn in enumerate(fa1): # Checking if any of the part names correspond to the boundary
            if pn == 'COLD':
                flag = 1
            if pn == 'HOT':
                flag = 1
        if flag == 0: # faces which do not belong to the boundary
            Acv2n_int[i,np.unique(cvofa[faono[i]])] = 1./nncv # Averaging surrounding CV termperatures and storing them in Tn. Note that np.unique ensures that each CV is accounted for only once        
    n = 1 # Counter for no. of iterations
     
    while n <= 500:
        
        # Updating CV centre values 
        Tcv = An2cv.dot(Tn)
        
        #Updating node values based on CV centres
        Tn_int = Acv2n_int.dot(Tcv)        
        
        #Assuming that All boundary nodes are loacted towards the very end of Tn
        Tn[0:Tn_int.size] = Tn_int        
        
#        print "Iteration: %d" %(n)
        n += 1 # Coded for 10000 iterations
    
    stop_time2 = time.time() - start_time2    
    print "Time required to execute matrix solve for Mesh C: %2.10f" %(stop_time2)


    start_time1 = time.time()  # Time Starts
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    cold_bc = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'COLD'
    hot_bc = np.where(partofa1=='HOT')   # Vectorized approach to find face belonging to part 'HOT'
    
    #Temperature initialized at nodes and CVs
    Tn = np.zeros(nno)    
    
    # Initializing values for hot and cold boundary nodes
    Tn[np.unique(noofa[cold_bc])] = 300 
    Tn[np.unique(noofa[hot_bc])] = 500
    
    # Initializing CV temperatures
    Tcv = np.zeros(ncv)
    
    n = 1 # Counter for no. of iterations
    
    while n <= 500:
        # Looping over each CV
        for i in np.arange(ncv):
            Tcv[i] = np.average(Tn[np.unique(noofa[faocv[i]])]) # Averaging surrounding nodal termperatures and storing them in Tcv. Note that np.unique ensures that each node is accounted for only once
        for j in np.arange(nno):
            flag = 0 # flag to check if node is boundary node 
            fa1 = partofa1[faono[j]] # corresponding array of parts to which the faces containing the node belong to
            for k,pn in enumerate(fa1): # Checking if any of the part names correspond to the boundary
                if pn == 'COLD':
                    flag = 1
                if pn == 'HOT':
                    flag = 1
            if flag == 0: # faces which do not belong to the boundary
                Tn[j] = np.average(Tcv[np.unique(cvofa[faono[j]])]) # Averaging surrounding CV termperatures and storing them in Tn. Note that np.unique ensures that each CV is accounted for only once
#        print "Iteration: %d" %(n)
        n += 1 # Coded for 10000 iterations
    stop_time1 = time.time() - start_time1    
    print "Time required to execute for looping for Mesh C: %2.10f" %(stop_time1)

    #Mesh4
    print "Mesh D"
    icemcfd_project_folder = './'
#    filename = '/mesh_set_1/heated_rod_ncv=103.msh'
    filename = 'mesh_set/Mesh4.msh'
    figure_folder = "../report/"

    mshfile_fullpath = icemcfd_project_folder + filename

    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)
    start_time2 = time.time()
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    cold_bc = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'COLD'
    hot_bc = np.where(partofa1=='HOT')   # Vectorized approach to find face belonging to part 'HOT'
    solid = np.where(partofa1=='SOLID')
    
    #Temperature initialized at nodes and CVs
    Tn = np.zeros(nno)    
    
    # Initializing values for hot and cold boundary nodes
    Tn[np.unique(noofa[cold_bc])] = 300 
    Tn[np.unique(noofa[hot_bc])] = 500
    
    # Initializing CV temperatures
    Tcv = np.zeros(ncv)
    
    # Node-to-CV conversion sparse matrix
    An2cv = scysparse.csr_matrix((ncv,nno),dtype="float64")
    
    # CV-to-node conversion matrix for internal nodes. Note that boundary nodes will be treated within the matrix itself.
    Acv2n_int = scysparse.csr_matrix((nno-np.unique(noofa[cold_bc]).size-np.unique(noofa[hot_bc]).size,ncv),dtype="float64")  # Only takes interior points
    
    for i in np.arange(ncv):
        
        nnn = np.unique(noofa[faocv[i]]).size # Gives number of neighbouring nodes, for calculating weight for each position = 1/no. of surrounding nodes
        An2cv[i,np.unique(noofa[faocv[i]])] = 1./nnn
    # This assumes that boundary nodes are numbered the towards the very end. If not, then God help you!!
    for i in np.arange(nno):
        nncv = np.unique(cvofa[faono[i]]).size  # No. of neighbouring CVs of the ndoe
        flag = 0 # flag to check if node is boundary node 
        fa1 = partofa1[faono[i]] # corresponding array of parts to which the faces containing the node belong to
        for k,pn in enumerate(fa1): # Checking if any of the part names correspond to the boundary
            if pn == 'COLD':
                flag = 1
            if pn == 'HOT':
                flag = 1
        if flag == 0: # faces which do not belong to the boundary
            Acv2n_int[i,np.unique(cvofa[faono[i]])] = 1./nncv # Averaging surrounding CV termperatures and storing them in Tn. Note that np.unique ensures that each CV is accounted for only once        
    n = 1 # Counter for no. of iterations
     
    while n <= 500:
        
        # Updating CV centre values 
        Tcv = An2cv.dot(Tn)
        
        #Updating node values based on CV centres
        Tn_int = Acv2n_int.dot(Tcv)        
        
        #Assuming that All boundary nodes are loacted towards the very end of Tn
        Tn[0:Tn_int.size] = Tn_int        
        
#        print "Iteration: %d" %(n)
        n += 1 # Coded for 10000 iterations
    
    stop_time2 = time.time() - start_time2    
    print "Time required to execute matrix solve for Mesh D: %2.10f" %(stop_time2)


    start_time1 = time.time()  # Time Starts
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    cold_bc = np.where(partofa1=='COLD')   # Vectorized approach to find face belonging to part 'COLD'
    hot_bc = np.where(partofa1=='HOT')   # Vectorized approach to find face belonging to part 'HOT'
    
    #Temperature initialized at nodes and CVs
    Tn = np.zeros(nno)    
    
    # Initializing values for hot and cold boundary nodes
    Tn[np.unique(noofa[cold_bc])] = 300 
    Tn[np.unique(noofa[hot_bc])] = 500
    
    # Initializing CV temperatures
    Tcv = np.zeros(ncv)
    
    n = 1 # Counter for no. of iterations
    
    while n <= 500:
        # Looping over each CV
        for i in np.arange(ncv):
            Tcv[i] = np.average(Tn[np.unique(noofa[faocv[i]])]) # Averaging surrounding nodal termperatures and storing them in Tcv. Note that np.unique ensures that each node is accounted for only once
        for j in np.arange(nno):
            flag = 0 # flag to check if node is boundary node 
            fa1 = partofa1[faono[j]] # corresponding array of parts to which the faces containing the node belong to
            for k,pn in enumerate(fa1): # Checking if any of the part names correspond to the boundary
                if pn == 'COLD':
                    flag = 1
                if pn == 'HOT':
                    flag = 1
            if flag == 0: # faces which do not belong to the boundary
                Tn[j] = np.average(Tcv[np.unique(cvofa[faono[j]])]) # Averaging surrounding CV termperatures and storing them in Tn. Note that np.unique ensures that each CV is accounted for only once
#        print "Iteration: %d" %(n)
        n += 1 # Coded for 10000 iterations
    stop_time1 = time.time() - start_time1    
    print "Time required to execute for looping for Mesh D: %2.10f" %(stop_time1)    
    
    
    
if p3:
    
    e_RMS = np.zeros(4) # RMS Error
    NCV = np.zeros(4)
    
    #Mesh1
    print "Mesh A"
    icemcfd_project_folder = './'
#    filename = '/mesh_set_1/heated_rod_ncv=103.msh'
    filename = 'mesh_set/Mesh1.msh'
    figure_folder = "../report/"

    mshfile_fullpath = icemcfd_project_folder + filename

    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)    
    
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    NCV[0] = ncv
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    
    u = (xy_no[:,1])*(xy_no[:,0]**2) #x-component of velocity
    v = -(xy_no[:,0])*(xy_no[:,1]**2) #y-component of velocity
    
    name = "Quiver_Plot_for_velocity _on_Mesh_A.pdf"
    figure_name = figure_folder + name
    figwidth       = 10
    figheight      = 8
    lineWidth      = 3
    textFontSize   = 10
    gcafontSize    = 14
    fig = plt.figure(0, figsize=(figwidth,figheight))
    plt.quiver(xy_no[:,0],xy_no[:,1],u,v)
    print "Saving figure: "+ figure_name
    plt.savefig(figure_name)
    plt.close()
    
    NORMAL = [] #blank normal array to be filled up
    AREA = np.zeros(ncv)
    #Pre-processing and finding normals over all CVs
    for i in np.arange(ncv):
        nocv = noofa[faocv[i]]   # Nodal pairs for each face of the CV
        face_co = xy_fa[faocv[i]] # Face centroids of each face of CV
        check_vecs = face_co - xy_cv[i] #Vectors from CV centre to face centre
        par_x = xy_no[nocv[:,1],0] - xy_no[nocv[:,0],0] #x-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        par_y = xy_no[nocv[:,1],1] - xy_no[nocv[:,0],1] #y-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        normal_fa = np.c_[-par_y,par_x]  #Defining normal vector to faces. Convention, normal is 90* clock-wise.
        dir_check = normal_fa[:,0]*check_vecs[:,0] + normal_fa[:,1]*check_vecs[:,1] # Checks if normal_fa is aligned in the same direction as check_vecs.
        normal_fa[np.where(dir_check<0)] = -normal_fa[np.where(dir_check<0)] # Flips sign of components in normal_fa where the dot product i.e. dir_check is negative
        NORMAL.append(normal_fa) # Spits out all normals indexed by Cvs
        #Calculating areas of CV assuming rectangles or triangles
        if np.size(faocv[i]) == 3:
            area_cv = np.abs(0.5*((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])))
            AREA[i] = area_cv
            
        if np.size(faocv[i]) == 4:
            area_cv = max(np.abs((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])),np.abs((par_x[0]*par_y[2]) - (par_x[2]*par_y[0])))
            AREA[i] = area_cv
        
    U_avg = 0.5*(u[noofa[np.arange(nfa)][:,0]] + u[noofa[np.arange(nfa)][:,1]]) # Vectorised average of u-velocities
    V_avg = 0.5*(v[noofa[np.arange(nfa)][:,0]] + v[noofa[np.arange(nfa)][:,1]]) # Vectorised average of u-velocities
    
    #Part 3a
    DIVERGENCE_1 = np.zeros(ncv)
    for j in np.arange(ncv):
        normal = NORMAL[j] #Normals for that CV
        U_CV_avg = U_avg[faocv[j]] #Average u-velocities for CV
        V_CV_avg = V_avg[faocv[j]] ##Average u-velocities for CV
        DIVERGENCE_1[j] = ((U_CV_avg.dot(normal[:,0])) + (V_CV_avg.dot(normal[:,1])))/AREA[j]
        
    e_RMS[0] = np.sqrt(sum(np.multiply(DIVERGENCE_1,AREA))**2/ncv)
        
    plot_data.plot_data(xy_cv[:,0],xy_cv[:,1],DIVERGENCE_1,"Flooded_Contour_of_Divergence_Mesh_A.pdf")
    
    #Part 3b
    Dx_n2cv = scysparse.csr_matrix((ncv,nno),dtype="float64") #Creating x part of operator
    Dy_n2cv = scysparse.csr_matrix((ncv,nno),dtype="float64") #Creating y part of operator
    
    DIVERGENCE_2 = np.zeros(ncv) #Divergence stored here
    
    for jj in np.arange(ncv):
        normal = NORMAL[jj] #Normals of the CV
        nocv = noofa[faocv[jj]] #Finding nodes in order of faces
        #Works as there are utmost 4 nodes right now. Dont know how slow it will be for higher order element shapes
        for ii,nn in enumerate(nocv[:,0]):
            Dx_n2cv[jj,nn] += 0.5*normal[ii,0]/AREA[jj]
            Dy_n2cv[jj,nn] += 0.5*normal[ii,1]/AREA[jj]
            
        for ii,nn in enumerate(nocv[:,1]):
            Dx_n2cv[jj,nn] += 0.5*normal[ii,0]/AREA[jj]
            Dy_n2cv[jj,nn] += 0.5*normal[ii,1]/AREA[jj]
            
            
    DIVERGENCE_2 = Dx_n2cv.dot(u) + Dy_n2cv.dot(v)
    
    plt.spy(Dx_n2cv)
    name = "Spy plot of Dx_n2cv_Mesh_A.pdf"
    figure_name = figure_folder + name
    print "Saving figure: "+ figure_name
    plt.savefig(figure_name)
    plt.close()
    
    plt.spy(Dy_n2cv)
    name = "Spy plot of Dy_n2cv_Mesh_A.pdf"
    figure_name = figure_folder + name
    print "Saving figure: "+ figure_name
    plt.savefig(figure_name)
    plt.close()
    
    
    #Mesh2
    print "Mesh B"
    icemcfd_project_folder = './'
#    filename = '/mesh_set_1/heated_rod_ncv=103.msh'
    filename = 'mesh_set/Mesh2.msh'
    figure_folder = "../report/"

    mshfile_fullpath = icemcfd_project_folder + filename

    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)    
    
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    NCV[1] = ncv
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    
    u = xy_no[:,1]*(xy_no[:,0]**2) #x-component of velocity
    v = -xy_no[:,0]*(xy_no[:,1]**2) #y-component of velocity
    
    NORMAL = [] #blank normal array to be filled up
    AREA = np.zeros(ncv)
    #Pre-processing and finding normals over all CVs
    for i in np.arange(ncv):
        nocv = noofa[faocv[i]]   # Nodal pairs for each face of the CV
        face_co = xy_fa[faocv[i]] # Face centroids of each face of CV
        check_vecs = face_co - xy_cv[i] #Vectors from CV centre to face centre
        par_x = xy_no[nocv[:,1],0] - xy_no[nocv[:,0],0] #x-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        par_y = xy_no[nocv[:,1],1] - xy_no[nocv[:,0],1] #y-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        normal_fa = np.c_[-par_y,par_x]  #Defining normal vector to faces. Convention, normal is 90* clock-wise.
        dir_check = normal_fa[:,0]*check_vecs[:,0] + normal_fa[:,1]*check_vecs[:,1] # Checks if normal_fa is aligned in the same direction as check_vecs.
        normal_fa[np.where(dir_check<0)] = -normal_fa[np.where(dir_check<0)] # Flips sign of components in normal_fa where the dot product i.e. dir_check is negative
        NORMAL.append(normal_fa) # Spits out all normals indexed by Cvs
        #Calculating areas of CV assuming rectangles or triangles
        if np.size(faocv[i]) == 3:
            area_cv = np.abs(0.5*((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])))
            AREA[i] = area_cv
            
        if np.size(faocv[i]) == 4:
            area_cv = max(np.abs((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])),np.abs((par_x[0]*par_y[2]) - (par_x[2]*par_y[0])))
            AREA[i] = area_cv
        
    U_avg = 0.5*(u[noofa[np.arange(nfa)][:,0]] + u[noofa[np.arange(nfa)][:,1]]) # Vectorised average of u-velocities
    V_avg = 0.5*(v[noofa[np.arange(nfa)][:,0]] + v[noofa[np.arange(nfa)][:,1]]) # Vectorised average of u-velocities
    
    #Part 3a
    DIVERGENCE_1 = np.zeros(ncv)
    for j in np.arange(ncv):
        normal = NORMAL[j] #Normals for that CV
        U_CV_avg = U_avg[faocv[j]] #Average u-velocities for CV
        V_CV_avg = V_avg[faocv[j]] ##Average u-velocities for CV
        DIVERGENCE_1[j] = ((U_CV_avg.dot(normal[:,0])) + (V_CV_avg.dot(normal[:,1])))/AREA[j]
        
    e_RMS[1] = np.sqrt(sum(np.multiply(DIVERGENCE_1,AREA))**2/ncv)
        
    plot_data.plot_data(xy_cv[:,0],xy_cv[:,1],DIVERGENCE_1,"Flooded_Contour_of_Divergence_Mesh_B.pdf")
    
    #Part 3b
    Dx_n2cv = scysparse.csr_matrix((ncv,nno),dtype="float64") #Creating x part of operator
    Dy_n2cv = scysparse.csr_matrix((ncv,nno),dtype="float64") #Creating y part of operator
    
    DIVERGENCE_2 = np.zeros(ncv) #Divergence stored here
    
    for jj in np.arange(ncv):
        normal = NORMAL[jj] #Normals of the CV
        nocv = noofa[faocv[jj]] #Finding nodes in order of faces
        #Works as there are utmost 4 nodes right now. Dont know how slow it will be for higher order element shapes
        for ii,nn in enumerate(nocv[:,0]):
            Dx_n2cv[jj,nn] += 0.5*normal[ii,0]/AREA[jj]
            Dy_n2cv[jj,nn] += 0.5*normal[ii,1]/AREA[jj]
            
        for ii,nn in enumerate(nocv[:,1]):
            Dx_n2cv[jj,nn] += 0.5*normal[ii,0]/AREA[jj]
            Dy_n2cv[jj,nn] += 0.5*normal[ii,1]/AREA[jj]
            
            
    DIVERGENCE_2 = Dx_n2cv.dot(u) + Dy_n2cv.dot(v)
    
    plt.spy(Dx_n2cv)
    name = "Spy plot of Dx_n2cv_Mesh_B.pdf"
    figure_name = figure_folder + name
    print "Saving figure: "+ figure_name
    plt.savefig(figure_name)
    plt.close()
    
    plt.spy(Dy_n2cv)
    name = "Spy plot of Dy_n2cv_Mesh_B.pdf"
    figure_name = figure_folder + name
    print "Saving figure: "+ figure_name
    plt.savefig(figure_name)
    plt.close()
    
    #Mesh3
    print "Mesh C"
    icemcfd_project_folder = './'
#    filename = '/mesh_set_1/heated_rod_ncv=103.msh'
    filename = 'mesh_set/Mesh3.msh'
    figure_folder = "../report/"

    mshfile_fullpath = icemcfd_project_folder + filename

    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)    
    
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    NCV[2] = ncv
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    
    u = xy_no[:,1]*(xy_no[:,0]**2) #x-component of velocity
    v = -xy_no[:,0]*(xy_no[:,1]**2) #y-component of velocity
    
    NORMAL = [] #blank normal array to be filled up
    AREA = np.zeros(ncv)
    #Pre-processing and finding normals over all CVs
    for i in np.arange(ncv):
        nocv = noofa[faocv[i]]   # Nodal pairs for each face of the CV
        face_co = xy_fa[faocv[i]] # Face centroids of each face of CV
        check_vecs = face_co - xy_cv[i] #Vectors from CV centre to face centre
        par_x = xy_no[nocv[:,1],0] - xy_no[nocv[:,0],0] #x-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        par_y = xy_no[nocv[:,1],1] - xy_no[nocv[:,0],1] #y-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        normal_fa = np.c_[-par_y,par_x]  #Defining normal vector to faces. Convention, normal is 90* clock-wise.
        dir_check = normal_fa[:,0]*check_vecs[:,0] + normal_fa[:,1]*check_vecs[:,1] # Checks if normal_fa is aligned in the same direction as check_vecs.
        normal_fa[np.where(dir_check<0)] = -normal_fa[np.where(dir_check<0)] # Flips sign of components in normal_fa where the dot product i.e. dir_check is negative
        NORMAL.append(normal_fa) # Spits out all normals indexed by Cvs
        #Calculating areas of CV assuming rectangles or triangles
        if np.size(faocv[i]) == 3:
            area_cv = np.abs(0.5*((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])))
            AREA[i] = area_cv
            
        if np.size(faocv[i]) == 4:
            area_cv = max(np.abs((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])),np.abs((par_x[0]*par_y[2]) - (par_x[2]*par_y[0])))
            AREA[i] = area_cv
        
    U_avg = 0.5*(u[noofa[np.arange(nfa)][:,0]] + u[noofa[np.arange(nfa)][:,1]]) # Vectorised average of u-velocities
    V_avg = 0.5*(v[noofa[np.arange(nfa)][:,0]] + v[noofa[np.arange(nfa)][:,1]]) # Vectorised average of u-velocities
    
    #Part 3a
    DIVERGENCE_1 = np.zeros(ncv)
    for j in np.arange(ncv):
        normal = NORMAL[j] #Normals for that CV
        U_CV_avg = U_avg[faocv[j]] #Average u-velocities for CV
        V_CV_avg = V_avg[faocv[j]] ##Average u-velocities for CV
        DIVERGENCE_1[j] = ((U_CV_avg.dot(normal[:,0])) + (V_CV_avg.dot(normal[:,1])))/AREA[j]
        
    e_RMS[2] = np.sqrt(sum(np.multiply(DIVERGENCE_1,AREA))**2/ncv)
        
    plot_data.plot_data(xy_cv[:,0],xy_cv[:,1],DIVERGENCE_1,"Flooded_Contour_of_Divergence_Mesh_C.pdf")
    
    #Part 3b
    Dx_n2cv = scysparse.csr_matrix((ncv,nno),dtype="float64") #Creating x part of operator
    Dy_n2cv = scysparse.csr_matrix((ncv,nno),dtype="float64") #Creating y part of operator
    
    DIVERGENCE_2 = np.zeros(ncv) #Divergence stored here
    
    for jj in np.arange(ncv):
        normal = NORMAL[jj] #Normals of the CV
        nocv = noofa[faocv[jj]] #Finding nodes in order of faces
        #Works as there are utmost 4 nodes right now. Dont know how slow it will be for higher order element shapes
        for ii,nn in enumerate(nocv[:,0]):
            Dx_n2cv[jj,nn] += 0.5*normal[ii,0]/AREA[jj]
            Dy_n2cv[jj,nn] += 0.5*normal[ii,1]/AREA[jj]
            
        for ii,nn in enumerate(nocv[:,1]):
            Dx_n2cv[jj,nn] += 0.5*normal[ii,0]/AREA[jj]
            Dy_n2cv[jj,nn] += 0.5*normal[ii,1]/AREA[jj]
            
            
    DIVERGENCE_2 = Dx_n2cv.dot(u) + Dy_n2cv.dot(v)
    
    plot_data.plot_data(xy_cv[:,0],xy_cv[:,1],DIVERGENCE_1-DIVERGENCE_2,"Flooded_Contour_of_Difference_in_Divergence_Mesh_C.pdf")
    
    plt.spy(Dx_n2cv)
    name = "Spy plot of Dx_n2cv_Mesh_C.pdf"
    figure_name = figure_folder + name
    print "Saving figure: "+ figure_name
    plt.savefig(figure_name)
    plt.close()
    
    plt.spy(Dy_n2cv)
    name = "Spy plot of Dy_n2cv_Mesh_C.pdf"
    figure_name = figure_folder + name
    print "Saving figure: "+ figure_name
    plt.savefig(figure_name)
    plt.close()
    
    #Mesh4
    print "Mesh D"
    icemcfd_project_folder = './'
#    filename = '/mesh_set_1/heated_rod_ncv=103.msh'
    filename = 'mesh_set/Mesh4.msh'
    figure_folder = "../report/"

    mshfile_fullpath = icemcfd_project_folder + filename

    part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa = umesh_reader.read_unstructured_grid(mshfile_fullpath,node_reordering=True)    
    
    nno = xy_no.shape[0]  # No. of nodes
    ncv = xy_cv.shape[0]  # No. of CVs
    NCV[3] = ncv
    nfa = xy_fa.shape[0]  # No. of faces
    partofa1 = np.array(partofa) # Converting partofa to an array
    
    u = xy_no[:,1]*(xy_no[:,0]**2) #x-component of velocity
    v = -xy_no[:,0]*(xy_no[:,1]**2) #y-component of velocity
    
    NORMAL = [] #blank normal array to be filled up
    AREA = np.zeros(ncv)
    #Pre-processing and finding normals over all CVs
    for i in np.arange(ncv):
        nocv = noofa[faocv[i]]   # Nodal pairs for each face of the CV
        face_co = xy_fa[faocv[i]] # Face centroids of each face of CV
        check_vecs = face_co - xy_cv[i] #Vectors from CV centre to face centre
        par_x = xy_no[nocv[:,1],0] - xy_no[nocv[:,0],0] #x-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        par_y = xy_no[nocv[:,1],1] - xy_no[nocv[:,0],1] #y-component of vector parallel to face. Convention, 2nd point - 1st point in nocv
        normal_fa = np.c_[-par_y,par_x]  #Defining normal vector to faces. Convention, normal is 90* clock-wise.
        dir_check = normal_fa[:,0]*check_vecs[:,0] + normal_fa[:,1]*check_vecs[:,1] # Checks if normal_fa is aligned in the same direction as check_vecs.
        normal_fa[np.where(dir_check<0)] = -normal_fa[np.where(dir_check<0)] # Flips sign of components in normal_fa where the dot product i.e. dir_check is negative
        NORMAL.append(normal_fa) # Spits out all normals indexed by Cvs
        #Calculating areas of CV assuming rectangles or triangles
        if np.size(faocv[i]) == 3:
            area_cv = np.abs(0.5*((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])))
            AREA[i] = area_cv
            
        if np.size(faocv[i]) == 4:
            area_cv = max(np.abs((par_x[0]*par_y[1]) - (par_x[1]*par_y[0])),np.abs((par_x[0]*par_y[2]) - (par_x[2]*par_y[0])))
            AREA[i] = area_cv
        
    U_avg = 0.5*(u[noofa[np.arange(nfa)][:,0]] + u[noofa[np.arange(nfa)][:,1]]) # Vectorised average of u-velocities
    V_avg = 0.5*(v[noofa[np.arange(nfa)][:,0]] + v[noofa[np.arange(nfa)][:,1]]) # Vectorised average of u-velocities
    
    #Part 3a
    DIVERGENCE_1 = np.zeros(ncv)
    for j in np.arange(ncv):
        normal = NORMAL[j] #Normals for that CV
        U_CV_avg = U_avg[faocv[j]] #Average u-velocities for CV
        V_CV_avg = V_avg[faocv[j]] ##Average u-velocities for CV
        DIVERGENCE_1[j] = ((U_CV_avg.dot(normal[:,0])) + (V_CV_avg.dot(normal[:,1])))/AREA[j]
        
    e_RMS[3] = np.sqrt(sum(np.multiply(DIVERGENCE_1,AREA))**2/ncv)
        
    plot_data.plot_data(xy_cv[:,0],xy_cv[:,1],DIVERGENCE_1,"Flooded_Contour_of_Divergence_Mesh_D.pdf")
    
    #Part 3b
    Dx_n2cv = scysparse.csr_matrix((ncv,nfa),dtype="float64") #Creating x part of operator
    Dy_n2cv = scysparse.csr_matrix((ncv,nno),dtype="float64") #Creating y part of operator
    
    DIVERGENCE_2 = np.zeros(ncv) #Divergence stored here
    
    for jj in np.arange(ncv):
        normal = NORMAL[jj] #Normals of the CV
        nocv = noofa[faocv[jj]] #Finding nodes in order of faces
        #Works as there are utmost 4 nodes right now. Dont know how slow it will be for higher order element shapes
        for ii,nn in enumerate(nocv[:,0]):
            Dx_n2cv[jj,nn] += 0.5*normal[ii,0]/AREA[jj]
            Dy_n2cv[jj,nn] += 0.5*normal[ii,1]/AREA[jj]
            
        for ii,nn in enumerate(nocv[:,1]):
            Dx_n2cv[jj,nn] += 0.5*normal[ii,0]/AREA[jj]
            Dy_n2cv[jj,nn] += 0.5*normal[ii,1]/AREA[jj]
            
            
    DIVERGENCE_2 = Dx_n2cv.dot(u) + Dy_n2cv.dot(v)
    
    plt.spy(Dx_n2cv)
    name = "Spy plot of Dx_n2cv_Mesh_D.pdf"
    figure_name = figure_folder + name
    print "Saving figure: "+ figure_name
    plt.savefig(figure_name)
    plt.close()
    
    plt.spy(Dy_n2cv)
    name = "Spy plot of Dy_n2cv_Mesh_D.pdf"
    figure_name = figure_folder + name
    print "Saving figure: "+ figure_name
    plt.savefig(figure_name)
    plt.close()
    
    # RMS rror plot
    name = "RMS_Error_for_Divergence.pdf"
    figure_name = figure_folder + name
    figwidth       = 10
    figheight      = 8
    lineWidth      = 3
    textFontSize   = 10
    gcafontSize    = 14
    fig = plt.figure(0, figsize=(figwidth,figheight))
    plt.loglog(NCV,e_RMS,'-k',label="RMS Error")
    plt.loglog(NCV,NCV**-1,'--r',label='Order 1')
    plt.loglog(NCV,NCV**-2,'--b',label='Order 2')
    plt.loglog(NCV,NCV**-3,'--g',label='Order 3')
    plt.xlabel("No. of CVs")
    plt.ylabel(r"RMS Error")
    plt.legend(loc='best')
    print "Saving figure: "+ figure_name
    plt.savefig(figure_name)
    plt.close()
        
# set_trace(), uncommenting this line pauses the code here, just like keyboard in Matlab

#####################################################
########## Plot Grid Labels / Connectivity ##########
#####################################################
#
#fig_width = 30
#fig_height = 17
#textFontSize   = 15
#gcafontSize    = 32
#lineWidth      = 2
#
#Plot_Node_Labels = True
#Plot_Face_Labels = True
#Plot_CV_Labels   = True
#
## the following enables LaTeX typesetting, which will cause the plotting to take forever..
## from matplotlib import rc as matplotlibrc
## matplotlibrc('text.latex', preamble='\usepackage{color}')
## matplotlibrc('text',usetex=True)
## matplotlibrc('font', family='serif')
#
#mgplx = 0.05*np.abs(max(xy_no[:,0])-min(xy_no[:,0]))
#mgply = 0.05*np.abs(max(xy_no[:,1])-min(xy_no[:,1]))
#xlimits = [min(xy_no[:,0])-mgplx,max(xy_no[:,0])+mgplx]
#ylimits = [min(xy_no[:,1])-mgply,max(xy_no[:,1])+mgply]
#
#fig = plt.figure(0,figsize=(fig_width,fig_height))
#ax = fig.add_subplot(111)
#ax.plot(xy_no[:,0],xy_no[:,1],'o',markersize=5,markerfacecolor='k')
#
#node_color = 'k'
#centroid_color = 'r'
#
#for inos_of_fa in noofa:
#   ax.plot(xy_no[inos_of_fa,0], xy_no[inos_of_fa,1], 'k-', linewidth = lineWidth)
#
#if Plot_Face_Labels:
#  nfa = xy_fa.shape[0] # number of faces
#  faces_indexes = range(0,nfa)
#  for x_fa,y_fa,ifa in zip(xy_fa[:,0],xy_fa[:,1],faces_indexes):
#    ax.text(x_fa,y_fa,repr(ifa),transform=ax.transData,color='k',
#        verticalalignment='center',horizontalalignment='center',fontsize=textFontSize )
#
#if Plot_Node_Labels:
#  nno = xy_no.shape[0] # number of nodes
#  node_indexes = range(0,nno)
#  for xn,yn,ino in zip(xy_no[:,0],xy_no[:,1],node_indexes):
#    ax.text(xn,yn,repr(ino),transform=ax.transData,color='r',
#        verticalalignment='top',horizontalalignment='left',fontsize=textFontSize )
#
#if Plot_CV_Labels:
#  ncv = xy_cv.shape[0]  # number of control volumes
#  cv_indexes = range(0,ncv)
#  for xcv,ycv,icv in zip(xy_cv[:,0],xy_cv[:,1],cv_indexes):
#    ax.text(xcv,ycv,repr(icv),transform=ax.transData,color='b',
#        verticalalignment='top',horizontalalignment='left',fontsize=textFontSize )
#
#ax.axis('equal')
#ax.set_xlim(xlimits)
#ax.set_ylim(ylimits)
#ax.set_xlabel(r'$x$',fontsize=1.5*gcafontSize)
#ax.set_ylabel(r'$y$',fontsize=1.5*gcafontSize)
#plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
#plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
#fig_name = filename.split('.')[0]+'.pdf'
#plt.savefig(fig_name)

#set_trace()
