import numpy as np
import math
import os,sys
import matplotlib.pyplot as plt
    
def get_coor(line):
    '''Return coordintes list from coordinates linefrom xyz file.'''
    coor=[]
    for i in range(1,4):
        coor.append(float(line.split()[i]))
    return coor 

def get_rdf(atoms_array,N_A,d_cutoff):
    '''return atoms distribution(rdf).'''
    b=atoms_array[:N_A]
    c = b[:,np.newaxis] - atoms_array
    d = np.sqrt(np.sum(c**2,axis=2)).flatten()
    e = d[d<d_cutoff]
    return e

def get_Neighbor(array_atoms,array_all,d_cutoff):
    '''Return the nearest_atom_number list of an atom list.'''
    NN_list=np.zeros(array_atoms.shape[0],)
    diff=array_atoms[:,np.newaxis] - array_all
    dis_array=np.sqrt(np.sum(diff**2,axis=2))
    for j in range(dis_array.shape[0]):
        for k in range(dis_array.shape[1]):
            if dis_array[j,k]<d_cutoff:
                NN_list[j]+=1
    return NN_list

def wrap_atom(center_atom,check_atom,zmax,zmin,r):
    '''Deal with th boundary condition. Wrap the atom back when generate a sphere.'''
    vec_dif=center_atom-check_atom
    z_len=zmax-zmin
    mid=(zmax+zmin)/2
    if np.abs(vec_dif[2])+r>z_len and center_atom[2]<=mid:
        check_atom[2]-=z_len
    if np.abs(vec_dif[2])+r>z_len and center_atom[2]>=mid:
        check_atom[2]+=z_len
    return check_atom

def get_sphere_wrap(center_atom,all_atom,zmax,zmin,r):
    '''Generate a shere of dis r after wrapping atoms back.'''
    sphere_arr=np.zeros((all_atom.shape[0],3))
    arr_copy=all_atom.copy()
    delete_list=[]
    for i in range(all_atom.shape[0]):
        sphere_arr[i]=wrap_atom(center_atom,arr_copy[i],zmax,zmin,r)
        if np.linalg.norm(center_atom-arr_copy[i])>r:
            delete_list.append(i) 
    sphere_n= np.delete(sphere_arr,delete_list,0)
    return sphere_n

def get_sphere(center_atom,all_atom,r):
    '''Generate a shere of dis r without wrapping atoms back.'''
    arr_copy=all_atom.copy()
    delete_list=[]
    for i in range(all_atom.shape[0]):
        if np.linalg.norm(center_atom-all_atom[i])>r:
            delete_list.append(i) 
    sphere= np.delete(arr_copy,delete_list,0)
    return sphere
