#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 12:02:18 2018

@author: hif1000
"""

import numpy as np
import os

def initParams():
    nnParams = {
            'nL1': 50,
            'nL2': 30,
            'learningRate': 0.0001,
            'nEpoch': 10000,
            }
    
    featParams = {
            'n2b' : 12,
            'n3b' : 3,
            }
    
    featParams['nFeat'] = featParams['n2b'] + featParams['n3b']**3
    featParams['featA'] = np.zeros([1, featParams['nFeat']])
    featParams['featB'] = np.zeros([1, featParams['nFeat']])
    
    return nnParams, featParams


def getCoord(poscar):
    with open(poscar,'r') as p:
        nAtoms=0
        lattice = np.zeros((3,3))
        p.readline()
        p.readline()
        lattice[0,:] = np.array(p.readline().split(),dtype=float)
        lattice[1,:] = np.array(p.readline().split(),dtype=float)
        lattice[2,:] = np.array(p.readline().split(),dtype=float)
        p.readline()
        nAtoms = int(p.readline())
        p.readline()
        R = np.zeros((nAtoms,3))
        for i in range(nAtoms):
            R[i] = np.array(p.readline().split()[:3],dtype=float)
    return nAtoms, R.dot(lattice)-[10,10,10]

def getE(poscar):
    with open(poscar,'r') as p:
        E = np.array(p.readline().split(),dtype=float)
        return E[1]-E[0]-E[2]+0.5
    
def getCos(x, numBasis):
    nodes = np.linspace(-1,1,numBasis)
    y = x[:,np.newaxis] - nodes
    h = 2/(numBasis-1)
    zeroMask = (y ==0)
    y[np.abs(y)>h] = 0
    y[y!=0] = np.cos(y[y!=0]/h*np.pi)/2+0.5
    y[zeroMask] = 1
    y[np.abs(x)>1] = 0
    return y

def getFeat(nAtoms, coord,n2b,n3b):
    Rl = np.sqrt(np.sum(coord**2,axis=1))[1:]
    Dc = coord[1:,np.newaxis] - coord[1:]
    Dc = np.sqrt(np.sum(Dc**2,axis=2))
    yR = np.sum(getCos(Rl/4-1,n2b),axis=0)
    yD = np.zeros((nAtoms-1,nAtoms-1,n3b))
    yD[Dc!=0] = getCos(Dc[Dc!=0]/4-1,n3b)
    yD = np.sum(getCos(Rl/4-1,n3b)[:,np.newaxis,:,np.newaxis] * yD[:,:,np.newaxis,:],axis=0)
    yD = np.sum(getCos(Rl/4-1,n3b)[:,np.newaxis,:,np.newaxis] * yD[:,:,np.newaxis,:],axis=0)
    return np.concatenate([yR, yD.reshape(-1)])
            
def getAd(AdDir, featParams):
    
    nAd = 0
    for file in os.listdir(AdDir):
        nAd += 1
    
    AdFeat = np.zeros((nAd, featParams['nFeat']))
    AdEngy = np.zeros(nAd)
    
    iAd = 0
    for file in os.listdir(AdDir):
        numA,coord=getCoord(AdDir+"/"+file)
        AdEngy[iAd] = getE(AdDir+"/"+file)
        AdFeat[iAd] = getFeat(numA, coord, featParams['n2b'], featParams['n3b'])
        iAd += 1
    
    return nAd, AdFeat, AdEngy

def getDe(DeDir, featParams):
    nDe = 0
    for file in os.listdir(DeDir):
        nDe += 1
    
    DeFeat = np.zeros((nDe, featParams['nFeat']))
        
    iDe = 0
    for file in os.listdir(DeDir):
        numA,coord=getCoord(DeDir+"/"+file)
        DeFeat[iDe] = getFeat(numA, coord, featParams['n2b'], featParams['n3b'])
        iDe += 1

    return nDe, DeFeat