import os
import sys
import numpy as np 
import scipy as sp 
from pdb import set_trace
from matplotlib import rc as matplotlibrc
import binascii

def read_unstructured_grid(filename,node_reordering=False):
  
  # this reader requires files in .msh format, generated
  # from ICEMCFD by selecting Fluent solver, 2D, ASCII output
  # Grid needs to be unstructured (Pre-Mesh --> Convert to Unstructured)
  
  f = open(filename, 'r')
  
  xy_no = []; part_names = []
  reading_nodes = False
  reading_faces = False
  reading_lines = True
  cvofa_dict = {}
  noofa_dict = {}

  while reading_lines:

    line = f.readline().strip()
    
    #**********************************************************************#
    #*************************** READING NODES ****************************#
    #**********************************************************************#

    if 'Node Section' in line:
      next_line = '('
      while '(' in next_line:
        next_line = f.readline().strip()
        continue
      
      line = next_line
      reading_nodes = True

    if reading_nodes:
      if ')' in line:
        reading_nodes = False
        #reading_lines = False
      else:
        xy_no.append([np.float64(s) for s in line.split(' ')])

    #**********************************************************************#
    #*************************** READING FACES ****************************#
    #**********************************************************************#

    if ('faces' in line) or ('Faces' in line):
      part_name = line.split(' ')[-1].strip('")"')
      part_names.append(part_name)
      next_line = '('
      while '(' in next_line:
        next_line = f.readline().strip()
        continue
      
      line = next_line
      reading_faces = True
    if reading_faces:
      if ')' in line[:1]:
        reading_faces = False
      else:
        if not cvofa_dict.has_key(part_name):
          cvofa_dict[part_name] = []
          noofa_dict[part_name] = []
        next_line = '('
        temp = [int(s, 16) for s in line.split(' ')]
        noofa_dict[part_name].append(temp[0:2])
        cvofa_dict[part_name].append(temp[2:])

    #**********************************************************************#
    #*************************** CHECKING END OF FILE**********************#
    #**********************************************************************#

    if 'Zone Sections' in line:
      reading_lines = False
      reading_faces = False

  interior_part_name = part_names[0] # assuming it's the first one in the .msh file
  
  xy_no = np.array(xy_no) # this is already globally indexed
  
  face_counter = 0
  partofa = []
  noofa = np.empty((0,2),dtype='int64')
  cvofa = np.empty((0,2),dtype='int64')
  for part_name in part_names:
     nfaopart = len(noofa_dict[part_name])
     part_name_list = list([part_name]*nfaopart)
     partofa += part_name_list
     ## The following '-1' is a shift so that this becomes consistent with Python numbering
     noofa = np.vstack([noofa,np.array(noofa_dict[part_name],dtype='int64')-1])
     cvofa = np.vstack([cvofa,np.array(cvofa_dict[part_name],dtype='int64')-1])

  ncv = cvofa.max()+1 # cv's indexes start from 0, so the the number of cells is cvofa.max()+1
  nno = noofa.max()+1 # number of nodes
  nfa_total = len(partofa)

  # need to build faono
  faono = []
  for ino in range(0,nno):
    faono.append(np.where(noofa==ino)[0])

  ### Need to reorder the node indexing, which will affect 
  ### the following arrays: noofa[ifa],xy_no[ino],
  if node_reordering:

    faono_old = np.array(faono)
    noofa_old = np.array(noofa)
    xy_no_old = np.array(xy_no)

    oldno_of_newno = []
    for ino_old in range(0,nno):
      ### Does node touch the boundary?
      node_touches_boundary=False
      for ifa in faono_old[ino_old]:
        if not(partofa[ifa]==interior_part_name):
          node_touches_boundary = True
      ### Start recounting with internal nodes
      if node_touches_boundary==False:
        oldno_of_newno.append(ino_old)

    inos_old_ext = set(range(0,nno))-set(oldno_of_newno)

    for ino_old_ext in inos_old_ext:
      oldno_of_newno.append(ino_old_ext)
    oldno_of_newno = np.array(oldno_of_newno)

    # re-shuffling faono & xy_no
    faono=np.array(faono_old[oldno_of_newno])
    xy_no=np.array(xy_no_old[oldno_of_newno])
    # re-shuffling noofa
    noofa = []
    for ifa in range(0,nfa_total):
      ino0, = np.where(oldno_of_newno==noofa_old[ifa][0])
      ino1, = np.where(oldno_of_newno==noofa_old[ifa][1])
      noofa.append([ino0,ino1])
    noofa = np.squeeze(noofa)
  ## end of node reordering...
  ##############################
  
  ####### Derived quantities
  xy_fa = []
  for ifa in range(0,nfa_total):
    xy_nodes_of_face = noofa[ifa]
    try:
      xy_fa.append(np.mean(xy_no[xy_nodes_of_face],axis=0))
    except:
      set_trace()
  xy_fa = np.array(xy_fa)

  faocv = []
  xy_cv = []
  for icv in range(0,ncv):
    ifas_of_cv = np.where(cvofa==icv)[0]
    faocv.append(ifas_of_cv)
    nfa_of_cv = ifas_of_cv.size
    xy_cv_temp = 0.0
    for ifa in ifas_of_cv:
       xy_cv_temp += xy_fa[ifa]/np.float64(nfa_of_cv)
    xy_cv.append(xy_cv_temp)

  xy_cv = np.array(xy_cv)
  
  # nodes can be shared by two zones, so perhaps zoneono is not well defined
  return part_names, xy_no, xy_fa, xy_cv, noofa, cvofa, faono, faocv, partofa
  
