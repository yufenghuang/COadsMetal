#!/usr/bin/env python3

"""
This script is used to test all the functions in function.py using the  AuC_NP.xyz file.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import os,sys
from function import *

print("\n........Generate corrdinates array from xyz file........\n")
print("\n........Test get_coor Function........\n")
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
xyz_in.close()
print('System has {} Au atoms, and the coordinates have been stored in Au_array;'.format(Au_array.shape[0]))
print('System has {} C atoms, and the coordinates have been stored in C_array;'.format(C_array.shape[0]))
print('System totally has {} atoms, and the coordinates have been stored in AuC_array;'.format(AuC_array.shape[0]))
print('\nFuncton get_coor() test complete.\n')


print("\n........Test plot_rdf Function........\n")
dis_lst=get_rdf(Au_array,100,6)
plt.hist(dis_lst,bins=100)
print('The nearest neighbor of Au is ~3, and second ~4 and third ~5.')
print('\nFuncton plot_rdf() test complete.\n')

print("\n........Test get_Neighbor Function........\n")
Au_neighbor=get_Neighbor(Au_array[:2],Au_array,3.3)
print('Perfect Au has 12 nearest neighbor\n')
print('\nFuncton get_Neighbor() test complete.\n')


print("\n........Test wrap_atom() Function........\n")
new=wrap_atom(np.array([0,0,0]),np.array([10,10,10.1]),10.5,0,1)
print('Before Wraping the coordinte is [10,10,10.1], after warpping coordinate is\
[{},{},{}]'.format(str(new[0]),str(new[1]),str(new[2])))
print('\nFuncton wrap_atom() test complete.\n')

print("\n........Test get_sphere_wrap() and get_sphere() Function........\n")
zmax=np.amax(AuC_array,axis=1)[2]
zmin=np.amin(AuC_array,axis=1)[2]
sphere_nowrap=get_sphere(Au_array[0],Au_array,10)
sphere_wrap=get_sphere_wrap(Au_array[0],Au_array,zmax,zmin,10)
if np.array_equiv(sphere_nowrap,sphere_wrap):
    print('Test atom is not around the boundary.')
else:
    print('Test atom is around the boundary.')
print('\nFunction get_sphere() test complete.')
print('Funciton get_sphere_wrap() test complete.\n')
