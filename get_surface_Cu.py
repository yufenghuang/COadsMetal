#!/usr/bin/env python3

import os,sys
import numpy as np
import matplotlib.pyplot as plt
from function import *

#######################################################################################
#                                                                                     #
#                       Step I: Generate array for all atom groups from xyz file      #
#                                                                                     #
#######################################################################################
xyz_in=open('CuC_NP.xyz','r')
N_atom=int(xyz_in.readline())
line=xyz_in.readline()
Cu_lst=[]
C_lst=[]
CuC_lst=[]
for i in range(N_atom):
   line=xyz_in.readline()
   CuC_lst.append(get_coor(line))
   if line.split()[0]=='Cu':
       Cu_lst.append(get_coor(line))
   if line.split()[0]=='C':
       C_lst.append(get_coor(line))
C_array=np.array(C_lst)       
Cu_array=np.array(Cu_lst)
CuC_array=np.array(CuC_lst)

#######################################################################################
#                                                                                     #
#                     Step II: Romove Cu with 8A of Carbon Nanotube                   #
#                                                                                     #
#######################################################################################
Cu_T=Cu_array.copy()
delete_lst=[]
n_cycle=int(np.floor(Cu_array.shape[0]/100))
for i in range(n_cycle):    
    b=Cu_T[i*100:(i+1)*100]
    c = b[:,np.newaxis] - C_array
    d = np.sqrt(np.sum(c**2,axis=2))
    e=np.amin(d,axis=1)
    for j in range(e.shape[0]):
        if e[j]<8:
            delete_lst.append(i*100+j)
b_left=Cu_T[n_cycle*100:]
c_left=b_left[:,np.newaxis] - C_array
d_left= np.sqrt(np.sum((c_left)**2,axis=2))
e_left=np.amin(d_left,axis=1)
for j in range(e_left.shape[0]):
    if e_left[j]<8:
        delete_lst.append(n_cycle*100+j)
Cu_T= np.delete(Cu_T,delete_lst,0)

#######################################################################################
#                                                                                     #
#            Step III: Get surface atom and defects points by NN method               #
#                                                                                     #
#######################################################################################
surface_out=open('surface_temp.xyz','w')
m_cycle=int(np.floor(Cu_T.shape[0]/100))
sur_NN=[]
for i in range(m_cycle):    
    b=Cu_T[i*100:(i+1)*100]
    NN_list=get_Neighbor(b,Cu_array,3.3)
    for p in range(len(NN_list)):
        if NN_list[p]!=13:
           lsst=list(b[p])
           sur_NN.append(lsst)
           surface_out.write('Cu '+str(lsst[0])+' '+str(lsst[1])+\
                             ' '+str(lsst[2])+'\n')           
b_left=Cu_T[m_cycle*100:]
NN_left_list=get_Neighbor(b_left,Cu_array,3.0)
for q in range(len(NN_left_list)):
        if NN_left_list[q]!=13:
            lsst=list(b[q])
            sur_NN.append(lsst)
            surface_out.write('Cu '+str(lsst[0])+' '+\
                              str(lsst[1])+' '+str(lsst[2])+'\n') 
surface_out.close()
xyz_in.close()  

#######################################################################################
#                                                                                     #
#          Step IV: Remove the atoms around defects using surface_vector method       #
#                                                                                     #
#######################################################################################
surface_out=open('surface_final.xyz','w')
array=np.array(sur_NN).reshape(len(sur_NN),3)
for i in range(array.shape[0]):
    sphere=get_sphere(array[i],Cu_array,r)
    for j in range(sphere.shape[0]):
        if np.array_equiv(sphere[j],array[i]):
            cid=j
    sphere_c=sphere[cid]
    sphere_em=np.delete(sphere,cid,0)-sphere_c
    vector=-np.sum(sphere_em,axis=0)
    angle_lst=[]
    for j in range(sphere_em.shape[0]):
        COS=np.dot(sphere_em[j],vector)/np.linalg.norm(vector)/np.linalg.norm(sphere_em[j])
        angle_lst.append(np.arccos(COS)*180/np.pi)
    if min(angle_lst)>30:
        lsst=list(array[i])
        surface_out.write('Cu {:.2f} {:.2f} {:.2f}\n'.format(lsst[0],lsst[1],lsst[2])) 
surface_out.close()

#######################################################################################
#                                                                                     #
#                     Step IV: Remove the atoms around the boundary                   #
#                                                                                     #
#######################################################################################
infile=open('surface_finel.xyz','r')
outfile=open('surface_Cu.xyz','w')
Num_a=int(infile.readline())
line=infile.readline()
for i in range(Num_a):
    ele=infile.readline().split()
    if float(ele[3])<(zmax-10) and float(ele[3])>(zmin+10):
        outfile.write('Cu {} {} {}\n'.format(ele[1],ele[2],ele[3])) 
outfile.close()
infile.close()        
            
