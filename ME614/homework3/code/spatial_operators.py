import numpy as np
import scipy.sparse as scysparse
import sys
from pdb import set_trace as keyboard
import scipy.sparse as scysparse
import scipy.sparse.linalg as spysparselinalg
import scipy.linalg as scylinalg        # non-sparse linear algebra

############################################################
############################################################

def create_DivGrad_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask,boundary_conditions="Homogeneous Neumann"):
	# defaults to "Homogeneous Neumann"
	
     possible_boundary_conditions = ["Homogeneous Dirichlet","Homogeneous Neumann","periodic"]

     if not(boundary_conditions in possible_boundary_conditions):
         sys.exit("Boundary conditions need to be either: " +
                  repr(possible_boundary_conditions))

     # numbering with -1 means that it is not a fluid cell (i.e. either ghost cell or external)
     numbered_pressureCells = -np.ones(Xc.shape,dtype='int64') #Wont make a difference even if its replaced by Yc as both are 2d arrays of the same shape
     jj_C,ii_C = np.where(pressureCells_Mask==False)
     Np = len(jj_C)         # total number of pressure nodes, not necessarily equal to Nxc*Nyc
     numbered_pressureCells[jj_C,ii_C] = range(0,Np) # automatic numbering done via 'C' flattening
#     print numbered_pressureCells.shape
     
     inv_DxC = 1./Dxc[jj_C,ii_C]
     inv_DyC = 1./Dyc[jj_C,ii_C]

     inv_DxE = 1./(Xc[jj_C,ii_C+1]-Xc[jj_C,ii_C])
     inv_DyN = 1./(Yc[jj_C+1,ii_C]-Yc[jj_C,ii_C]);

     inv_DxW = 1./(Xc[jj_C,ii_C]-Xc[jj_C,ii_C-1])
     inv_DyS = 1./(Yc[jj_C,ii_C]-Yc[jj_C-1,ii_C])
     
     DivGrad = scysparse.csr_matrix((Np,Np),dtype="float64") # initialize with all zeros
	 
     iC = numbered_pressureCells[jj_C,ii_C]
     #print iC
     iE = numbered_pressureCells[jj_C,ii_C+1]
     #print iE
     iW = numbered_pressureCells[jj_C,ii_C-1]
     #print iW
     iS = numbered_pressureCells[jj_C-1,ii_C]
     #print iS
     iN = numbered_pressureCells[jj_C+1,ii_C]
     #print iN

     # consider pre-multiplying all of the weights by the local value of dx*dy

     # start by creating operator assuming homogeneous Neumann

     ## if east node is inside domain
     east_node_mask = (iE!=-1)
     ii_center = iC[east_node_mask]
     ii_east   = iE[east_node_mask]
     inv_dxc_central = inv_DxC[ii_center]
     inv_dxc_east    = inv_DxE[ii_center]
#     keyboard()
     DivGrad[ii_center,ii_east]   += inv_dxc_central*inv_dxc_east
     DivGrad[ii_center,ii_center] -= inv_dxc_central*inv_dxc_east
     
     ## if west node is inside domain
     west_node_mask = (iW!=-1)
     ii_center  = iC[west_node_mask]
     ii_west    = iW[west_node_mask]
     inv_dxc_central = inv_DxC[ii_center]
     inv_dxc_west    = inv_DxW[ii_center]
     DivGrad[ii_center,ii_west]   += inv_dxc_central*inv_dxc_west
     DivGrad[ii_center,ii_center] -= inv_dxc_central*inv_dxc_west

	 ## if north node is inside domain
     north_node_mask = (iN!=-1)
     ii_center  = iC[north_node_mask]
     ii_north   = iN[north_node_mask]
     inv_dyc_central  = inv_DyC[ii_center]
     inv_dyc_north    = inv_DyN[ii_center]
     DivGrad[ii_center,ii_north]   += inv_dyc_central*inv_dyc_north
     DivGrad[ii_center,ii_center]  -= inv_dyc_central*inv_dyc_north

      ## if south node is inside domain
     south_node_mask = (iS!=-1)
     ii_center  = iC[south_node_mask]
     ii_south   = iS[south_node_mask]
     inv_dyc_central  = inv_DyC[ii_center]
     inv_dyc_south    = inv_DyS[ii_center]
     DivGrad[ii_center,ii_south]   += inv_dyc_central*inv_dyc_south
     DivGrad[ii_center,ii_center]  -= inv_dyc_central*inv_dyc_south

	 # if Dirichlet boundary conditions are requested, need to modify operator
     if boundary_conditions == "Homogeneous Dirichlet":

		# for every east node that is 'just' outside domain
		east_node_mask = (iE==-1)&(iC!=-1)
		ii_center = iC[east_node_mask]
		inv_dxc_central = inv_DxC[ii_center]
		inv_dxc_east    = inv_DxE[ii_center]
		DivGrad[ii_center,ii_center]  -= 2.*inv_dxc_central*inv_dxc_east
		
		# for every west node that is 'just' outside domain
		west_node_mask = (iW==-1)&(iC!=-1)
		ii_center = iC[west_node_mask]
		inv_dxc_central = inv_DxC[ii_center]
		inv_dxc_west    = inv_DxW[ii_center]
		DivGrad[ii_center,ii_center]  -= 2.*inv_dxc_central*inv_dxc_west

		# for every north node that is 'just' outside domain
		north_node_mask = (iN==-1)&(iC!=-1)
		ii_center = iC[north_node_mask]
		inv_dyc_central  = inv_DyC[ii_center]
		inv_dyc_north    = inv_DyN[ii_center]
		DivGrad[ii_center,ii_center]  -= 2.*inv_dyc_central*inv_dyc_north

		# for every south node that is 'just' outside domain
		south_node_mask = (iS==-1)&(iC!=-1)
		ii_center = iC[south_node_mask]
		inv_dyc_central  = inv_DyC[ii_center]
		inv_dyc_south    = inv_DyS[ii_center]
		DivGrad[ii_center,ii_center]  -= 2.*inv_dyc_central*inv_dyc_south
  
     if boundary_conditions == "periodic":

		# for every east and west node that is 'just' outside domain
        east_node_mask = (iE==-1)&(iC!=-1)
        west_node_mask = (iW==-1)&(iC!=-1)
        ii_center = iC[east_node_mask]
        ii_center_west = iC[west_node_mask]
        inv_dxc_central = inv_DxC[ii_center]
        inv_dxc_east    = inv_DxE[ii_center]
        DivGrad[ii_center,ii_center]  -= inv_dxc_central*inv_dxc_east
        DivGrad[ii_center,ii_center_west] += inv_dxc_central*inv_dxc_east
        
        ii_center = iC[west_node_mask]
        ii_center_east = iC[east_node_mask]
        inv_dxc_central = inv_DxC[ii_center]
        inv_dxc_west    = inv_DxW[ii_center]
        DivGrad[ii_center,ii_center]  -= inv_dxc_central*inv_dxc_west
        DivGrad[ii_center,ii_center_east] += inv_dxc_central*inv_dxc_west

		# for every north and south node that is 'just' outside domain
        north_node_mask = (iN==-1)&(iC!=-1)
        south_node_mask = (iS==-1)&(iC!=-1)
        ii_center = iC[north_node_mask]
        ii_south_center = iC[south_node_mask]
        inv_dyc_central  = inv_DyC[ii_center]
        inv_dyc_north    = inv_DyN[ii_center]
        DivGrad[ii_center,ii_center]  -= inv_dyc_central*inv_dyc_north
        DivGrad[ii_center,ii_south_center] +=inv_dyc_central*inv_dyc_north
        
        ii_center = iC[south_node_mask]
        ii_north_center = iC[north_node_mask]
        inv_dyc_central  = inv_DyC[ii_center]
        inv_dyc_south    = inv_DyS[ii_center]
        DivGrad[ii_center,ii_center]  -= inv_dyc_central*inv_dyc_south
        DivGrad[ii_center,ii_north_center] +=inv_dyc_central*inv_dyc_south


     return DivGrad
     
def create_delx_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask,discretization,boundary_conditions):
	# does not defaults to "Homogeneous Neumann"
    possible_boundary_conditions = ["Homogeneous Dirichlet","periodic"]
    if not(boundary_conditions in possible_boundary_conditions):
        sys.exit("Boundary conditions need to be either: " +
        repr(possible_boundary_conditions))

     # numbering with -1 means that it is not a fluid cell (i.e. either ghost cell or external)
    numbered_pressureCells = -np.ones(Xc.shape,dtype='int64') #Wont make a difference even if its replaced by Yc as both are 2d arrays of the same shape
    jj_C,ii_C = np.where(pressureCells_Mask==False)
    Np = len(jj_C)         # total number of pressure nodes, not necessarily equal to Nxc*Nyc
    numbered_pressureCells[jj_C,ii_C] = range(0,Np) # automatic numbering done via 'C' flattening
#     print numbered_pressureCells.shape
     

    DxE = (Xc[jj_C,ii_C+1]-Xc[jj_C,ii_C])

    DxW = (Xc[jj_C,ii_C]-Xc[jj_C,ii_C-1])
    
     
    delx = scysparse.csr_matrix((Np,Np),dtype="float64") # initialize with all zeros
	 
    iC = numbered_pressureCells[jj_C,ii_C]
    #print iC
    iE = numbered_pressureCells[jj_C,ii_C+1]
    #print iE
    iW = numbered_pressureCells[jj_C,ii_C-1]
    #print iW
    
    #print iS
    
    #print iN

     # consider pre-multiplying all of the weights by the local value of dx*dy

     # start by creating operator assuming homogeneous Neumann

     ## if both east and west nodes are inside domain
    if discretization == "2nd-order-central":
        
        node_mask = (iE!=-1)&(iC!=-1)
        ii_center = iC[node_mask]
        ii_east   = iE[node_mask]
        dxc_west = DxW[ii_center]
        dxc_east = DxE[ii_center]
#     keyboard()
        delx[ii_center,ii_east]   += 1./(dxc_east+dxc_west)
        delx[ii_center,ii_center] += 0
        
        node_mask = (iW!=-1)&(iC!=-1)
        ii_center = iC[node_mask]
        ii_west   = iW[node_mask]
        dxc_west = DxW[ii_center]
        dxc_east = DxE[ii_center]
#     keyboard()
        delx[ii_center,ii_west]   += -1./(dxc_east+dxc_west)
        delx[ii_center,ii_center] += 0
    
    if discretization == "1st-order-upwind":
        
        #only need east points in the boundary
        node_mask = (iW!=-1)
        ii_center = iC[node_mask]
        ii_west   = iW[node_mask]
        dxc_west = DxW[ii_center]
#     keyboard()
        delx[ii_center,ii_west]   += -1./(dxc_west)
        delx[ii_center,ii_center] += 1/(dxc_west)
    
     
	 # if Dirichlet boundary conditions are requested, need to modify operator
    if boundary_conditions == "Homogeneous Dirichlet":


        if discretization == "2nd-order-central":
		# for every east node that is 'just' outside domain
            east_node_mask = (iE==-1)&(iC!=-1)
            ii_center = iC[east_node_mask]
            ii_west   = iW[east_node_mask]  
            dxc_east = DxE[ii_center]
            dxc_west = DxW[ii_center]
            delx[ii_center,ii_center]  += -1./(dxc_west+dxc_east)
            delx[ii_center,ii_west]  += -1./(dxc_west+dxc_east)
		
		# for every west node that is 'just' outside domain
            west_node_mask = (iW==-1)&(iC!=-1)
            ii_center = iC[west_node_mask]
            ii_east   = iE[west_node_mask]  
            dxc_east = DxE[ii_center]
            dxc_west = DxW[ii_center]
            delx[ii_center,ii_center]  += 1./(dxc_west+dxc_east)
            delx[ii_center,ii_east]  += 1./(dxc_west+dxc_east)
            
        if discretization == "1st-order-upwind":
            west_node_mask = (iW==-1)&(iC!=-1)
            ii_center = iC[west_node_mask]
            ii_west   = iW[west_node_mask]
            dxc_west = DxW[ii_center]
            delx[ii_center,ii_center]  += -2./(dxc_west)
            
            
    if boundary_conditions == "periodic":


        if discretization == "2nd-order-central":
		# for every east and west node that is 'just' outside domain
            east_node_mask = (iE==-1)&(iC!=-1)
            west_node_mask = (iW==-1)&(iC!=-1)
            ii_center = iC[east_node_mask]
            ii_west_center = iC[west_node_mask]  
            dxc_east = DxE[ii_center]
            dxc_west = DxW[ii_center]
            delx[ii_center,ii_center]  += 0
            delx[ii_center,ii_west_center]  += 1./(dxc_west+dxc_east)
            
            ii_center = iC[west_node_mask]
            ii_east_center = iC[east_node_mask]  
            dxc_east = DxE[ii_center]
            dxc_west = DxW[ii_center]
            delx[ii_center,ii_center]  += 0
            delx[ii_center,ii_east_center]  += -1./(dxc_west+dxc_east)
            
        if discretization == "1st-order-upwind":
            west_node_mask = (iW==-1)&(iC!=-1)
            east_node_mask = (iE==-1)&(iC!=-1)
            ii_center = iC[west_node_mask]
            ii_east_center   = iC[east_node_mask]
            dxc_west = DxW[ii_center]
            delx[ii_center,ii_center]  += -1./(dxc_west)
            delx[ii_center,ii_east_center]  += 1./(dxc_west)
            
    return delx


def create_dely_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask,discretization,boundary_conditions="Homogeneous Neumann"):
	# does not defaults to "Homogeneous Neumann"
    possible_boundary_conditions = ["Homogeneous Dirichlet","periodic"]
    if not(boundary_conditions in possible_boundary_conditions):
        sys.exit("Boundary conditions need to be either: " +
        repr(possible_boundary_conditions))

     # numbering with -1 means that it is not a fluid cell (i.e. either ghost cell or external)
    numbered_pressureCells = -np.ones(Xc.shape,dtype='int64') #Wont make a difference even if its replaced by Yc as both are 2d arrays of the same shape
    jj_C,ii_C = np.where(pressureCells_Mask==False)
    Np = len(jj_C)         # total number of pressure nodes, not necessarily equal to Nxc*Nyc
    numbered_pressureCells[jj_C,ii_C] = range(0,Np) # automatic numbering done via 'C' flattening
#     print numbered_pressureCells.shape
     

    DyN = (Yc[jj_C+1,ii_C]-Yc[jj_C,ii_C])

    DyS = (Yc[jj_C,ii_C]-Yc[jj_C-1,ii_C])
    
     
    dely = scysparse.csr_matrix((Np,Np),dtype="float64") # initialize with all zeros
	 
    iC = numbered_pressureCells[jj_C,ii_C]
    #print iC
    iN = numbered_pressureCells[jj_C+1,ii_C]
    #print iE
    iS = numbered_pressureCells[jj_C-1,ii_C]
    #print iW
    
    #print iS
    
    #print iN

     # consider pre-multiplying all of the weights by the local value of dx*dy

     # start by creating operator assuming homogeneous Neumann

     ## if both east and west nodes are inside domain
    if discretization == "2nd-order-central":
        
        node_mask = (iN!=-1)&(iC!=-1)
        ii_center = iC[node_mask]
        ii_north   = iN[node_mask]
        dyc_north = DyN[ii_center]
        dyc_south = DyS[ii_center]
#     keyboard()
        dely[ii_center,ii_north]   += 1./(dyc_north+dyc_south)
        dely[ii_center,ii_center] += 0
        
        node_mask = (iS!=-1)&(iC!=-1)
        ii_center = iC[node_mask]
        ii_south   = iS[node_mask]
        dyc_north = DyN[ii_center]
        dyc_south = DyS[ii_center]
#     keyboard()
        dely[ii_center,ii_south]   += -1./(dyc_north+dyc_south)
        dely[ii_center,ii_center] += 0
    
    if discretization == "1st-order-upwind":
        
        #only need east points in the boundary
        node_mask = (iS!=-1)
        ii_center = iC[node_mask]
        ii_south   = iS[node_mask]
        dyc_south = DyS[ii_center]
#     keyboard()
        dely[ii_center,ii_south]   += -1./(dyc_south)
        dely[ii_center,ii_center] += 1/(dyc_south)
    
     
	 # if Dirichlet boundary conditions are requested, need to modify operator
    if boundary_conditions == "Homogeneous Dirichlet":


        if discretization == "2nd-order-central":
		# for every north node that is 'just' outside domain
            north_node_mask = (iN==-1)&(iC!=-1)
            ii_center = iC[north_node_mask]
            ii_south   = iS[north_node_mask]  
            dyc_south = DyS[ii_center]
            dyc_north = DyN[ii_center]
            dely[ii_center,ii_center]  += -1./(dyc_south+dyc_north)
            dely[ii_center,ii_south]  += -1./(dyc_south+dyc_north)
		
		# for every south node that is 'just' outside domain
            south_node_mask = (iS==-1)&(iC!=-1)
            ii_center = iC[south_node_mask]
            ii_north   = iN[south_node_mask]  
            dyc_north = DyN[ii_center]
            dyc_south = DyS[ii_center]
            dely[ii_center,ii_center]  += 1./(dyc_south+dyc_north)
            dely[ii_center,ii_north]  += 1./(dyc_north+dyc_south)
            
        if discretization == "1st-order-upwind":
            south_node_mask = (iS==-1)&(iC!=-1)
            ii_center = iC[south_node_mask]
            ii_south   = iS[south_node_mask]
            dyc_south = DyS[ii_center]
            dely[ii_center,ii_center]  += -2./(dyc_south)
            dely[ii_center,ii_south]  += 0
            
    if boundary_conditions == "periodic":


        if discretization == "2nd-order-central":
		# for every north and south node that is 'just' outside domain
            north_node_mask = (iN==-1)&(iC!=-1)
            south_node_mask = (iS==-1)&(iC!=-1)
            ii_center = iC[north_node_mask]
            ii_south_center   = iC[south_node_mask]  
            dyc_south = DyS[ii_center]
            dyc_north = DyN[ii_center]
            dely[ii_center,ii_center]  += 0
            dely[ii_center,ii_south_center]  += 1./(dyc_south+dyc_north)
            
            ii_center = iC[south_node_mask]
            ii_north_center   = iC[north_node_mask]  
            dyc_south = DyS[ii_center]
            dyc_north = DyN[ii_center]
            dely[ii_center,ii_center]  += 0
            dely[ii_center,ii_north_center]  += -1./(dyc_south+dyc_north)
            
        if discretization == "1st-order-upwind":
            south_node_mask = (iS==-1)&(iC!=-1)
            north_node_mask = (iN==-1)&(iC!=-1)
            ii_center = iC[south_node_mask]
            ii_north_center   = iC[north_node_mask]
            dyc_south = DyS[ii_center]
            dely[ii_center,ii_center]  += -1./(dyc_south)
            dely[ii_center,ii_north_center]  += 1./(dyc_south)
    return dely

# IN this operator, unlike the others, the pressure differences are passed instead of the grid spacings
def create_divx_operator(Dxc,Dyc,Xc,Yc,Cells_Mask,discretization):
	# does not defaults to "Homogeneous Neumann"
#    possible_boundary_conditions = ["periodic"]
#    if not(boundary_conditions in possible_boundary_conditions):
#        sys.exit("Boundary conditions need to be: " +
#        repr(possible_boundary_conditions))

     # numbering with -1 means that it is not a fluid cell (i.e. either ghost cell or external)
    numbered_pressureCells = -np.ones(Xc.shape,dtype='int64') #Wont make a difference even if its replaced by Yc as both are 2d arrays of the same shape
    jj_C,ii_C = np.where(Cells_Mask==False)
    Np = len(jj_C)         # total number of pressure nodes, not necessarily equal to Nxc*Nyc
    numbered_pressureCells[jj_C,ii_C] = range(0,Np) # automatic numbering done via 'C' flattening
#     print numbered_pressureCells.shape
     

    DxE = (Xc[jj_C,ii_C+1]-Xc[jj_C,ii_C])

    divx = scysparse.csr_matrix((Np,Np),dtype="float64") # initialize with all zeros
	 
    iC = numbered_pressureCells[jj_C,ii_C]
    #print iC
    iE = numbered_pressureCells[jj_C,ii_C+1]
    
    iW = numbered_pressureCells[jj_C,ii_C-1]
    #print iE
    #print iW
    
    #print iS
    
    #print iN

     # consider pre-multiplying all of the weights by the local value of dx*dy

     # start by creating operator assuming homogeneous Neumann

     ## if both east and west nodes are inside domain

    if discretization == "2nd-order-central":
        
        #only need to check for east points in the boundary as divergence is calculated at pressure nodes
        node_mask = (iE!=-1)
        ii_center = iC[node_mask]
        ii_east   = iE[node_mask]
        dxc_east = DxE[ii_center]
#     keyboard()
        divx[ii_center,ii_east]   += 1./(dxc_east)
        divx[ii_center,ii_center] += -1/(dxc_east)
    
     
	 # if Dirichlet boundary conditions are requested, need to modify operator
    
            
    return divx[np.unique(divx.nonzero()[0])]


def create_divy_operator(Dxc,Dyc,Xc,Yc,pressureCells_Mask,discretization):
	# does not defaults to "Homogeneous Neumann"
#    possible_boundary_conditions = ["Homogeneous Dirichlet","periodic"]
#    if not(boundary_conditions in possible_boundary_conditions):
#        sys.exit("Boundary conditions need to be either: " +
#        repr(possible_boundary_conditions))

     # numbering with -1 means that it is not a fluid cell (i.e. either ghost cell or external)
    numbered_pressureCells = -np.ones(Xc.shape,dtype='int64') #Wont make a difference even if its replaced by Yc as both are 2d arrays of the same shape
    jj_C,ii_C = np.where(pressureCells_Mask==False)
    Np = len(jj_C)         # total number of pressure nodes, not necessarily equal to Nxc*Nyc
    numbered_pressureCells[jj_C,ii_C] = range(0,Np) # automatic numbering done via 'C' flattening
#     print numbered_pressureCells.shape
     

    DyN = (Yc[jj_C+1,ii_C]-Yc[jj_C,ii_C])
     
    divy = scysparse.csr_matrix((Np,Np),dtype="float64") # initialize with all zeros
	 
    iC = numbered_pressureCells[jj_C,ii_C]
    #print iC
    iN = numbered_pressureCells[jj_C+1,ii_C]
    iS = numbered_pressureCells[jj_C-1,ii_C]
    #print iE
    #print iW
    
    #print iS
    
    #print iN

     # consider pre-multiplying all of the weights by the local value of dx*dy

     # start by creating operator assuming homogeneous Neumann

     
    if discretization == "2nd-order-central":
        
        node_mask = (iN!=-1)
        ii_center = iC[node_mask]
        ii_north   = iN[node_mask]
        dyc_north = DyN[ii_center]
#     keyboard()
        divy[ii_center,ii_north]   = 1./(dyc_north)
        divy[ii_center,ii_center]   = -1./(dyc_north)
            
    return divy[np.unique(divy.nonzero()[0])]