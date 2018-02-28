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
xyz_in=open('AuC_NP.xyz','r')
N_atom=int(xyz_in.readline())
line=xyz_in.readline()
Au_lst=[]
C_lst=[]
AuC_lst=[]
for i in range(N_atom):
   line=xyz_in.readline()
   AuC_lst.append(get_coor(line))
   if line.split()[0]=='Au':
       Au_lst.append(get_coor(line))
   if line.split()[0]=='C':
       C_lst.append(get_coor(line))
C_array=np.array(C_lst)       
Au_array=np.array(Au_lst)
AuC_array=np.array(AuC_lst)

#######################################################################################
#                                                                                     #
#                     Step II: Romove Au with 8A of Carbon Nanotube                   #
#                                                                                     #
#######################################################################################
Au_T=Au_array.copy()
delete_lst=[]
n_cycle=int(np.floor(Au_array.shape[0]/100))
for i in range(n_cycle):    
    b=Au_T[i*100:(i+1)*100]
    c = b[:,np.newaxis] - C_array
    d = np.sqrt(np.sum(c**2,axis=2))
    e=np.amin(d,axis=1)
    for j in range(e.shape[0]):
        if e[j]<8:
            delete_lst.append(i*100+j)
b_left=Au_T[n_cycle*100:]
c_left=b_left[:,np.newaxis] - C_array
d_left= np.sqrt(np.sum((c_left)**2,axis=2))
e_left=np.amin(d_left,axis=1)
for j in range(e_left.shape[0]):
    if e_left[j]<8:
        delete_lst.append(n_cycle*100+j)
Au_T= np.delete(Au_T,delete_lst,0)

#######################################################################################
#                                                                                     #
#            Step III: Get surface atom and defects points by NN method               #
#                                                                                     #
#######################################################################################
surface_out=open('surface_temp.xyz','w')
m_cycle=int(np.floor(Au_T.shape[0]/100))
sur_NN=[]
for i in range(m_cycle):    
    b=Au_T[i*100:(i+1)*100]
    NN_list=get_Neighbor(b,Au_array,3.3)
    for p in range(len(NN_list)):
        if NN_list[p]!=13:
           lsst=list(b[p])
           sur_NN.append(lsst)
           surface_out.write('Au '+str(lsst[0])+' '+str(lsst[1])+\
                             ' '+str(lsst[2])+'\n')           
b_left=Au_T[m_cycle*100:]
NN_left_list=get_Neighbor(b_left,Au_array,3.3)
for q in range(len(NN_left_list)):
        if NN_left_list[q]!=13:
            lsst=list(b[q])
            sur_NN.append(lsst)
            surface_out.write('Au '+str(lsst[0])+' '+\
                              str(lsst[1])+' '+str(lsst[2])+'\n') 
surface_out.close()
xyz_in.close()  

#######################################################################################
#                                                                                     #
#          Step IV: Remove the atomd around defects using surface_vector method       #
#                                                                                     #
#######################################################################################
surface_out=open('surface_Au.xyz','w')
array=np.array(sur_NN).reshape(len(sur_NN),3)
for i in range(array.shape[0]):
    sphere=get_sphere(array[i],Au_array,r)
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
        surface_out.write('Au {:.2f} {:.2f} {:.2f}\n'.format(lsst[0],lsst[1],lsst[2])) 
surface_out.close()
        
     
        
        
            
