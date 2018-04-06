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
        if (p.readline().strip(" "))[0].upper() == "S":
            p.readline()
        R = np.zeros((nAtoms,3))
        for i in range(nAtoms):
            R[i] = np.array(p.readline().split()[:3],dtype=float)
    return nAtoms, R.dot(lattice)-[10,10,10]

def getNN(Rin, Rall, Rcut):
    return Rall[np.sum((Rin - Rall)**2,axis=1) < Rcut**2]

def loadMCxyz(xyzFile):
    with open(xyzFile, 'r') as file:
        nAtoms = int(file.readline())
        R = np.zeros((nAtoms, 3))
        isMetal = np.ones(nAtoms,dtype=int)
        file.readline()
        for i in range(nAtoms):
            line = file.readline().split()
            if line[0] == "C":
                isMetal[i] = 0
            R[i] = np.array(line[1:4], dtype=float)
    return isMetal, R

def loadXYZ(xyzFile):
    with open(xyzFile, 'r') as file:
        nAtoms = int(file.readline())
        R = np.zeros((nAtoms, 3))
        atomList = np.zeros(nAtoms, dtype=int)
        atomType = []
        
        file.readline()
        
        for i in range(nAtoms):
            line = file.readline().split()
            if line[0] not in atomType:
                atomType.append(line[0])
            atomList[i] = atomType.index(line[0])
            R[i] = np.array(line[1:4], dtype=float)
    return atomList, atomType, R

def removeC(R_M, R_C, Rc, chunkSize=1000):
    idxM = np.arange(len(R_M),dtype=int)
    
    nChunk = np.ceil(len(R_M)/chunkSize).astype(int)
    Rsplit = np.array_split(R_M, nChunk)
    idxSpl = np.array_split(idxM, nChunk)
    
    MnearC = np.zeros(len(R_M),dtype=bool)
        
    for i in range(nChunk):
        isMnearC = np.amin(np.sum(((Rsplit[i])[:,np.newaxis,:] - R_C)**2,axis=2),axis=1)<Rc**2
        MnearC[(idxSpl[i])[isMnearC]] = True
        print("Removing metal atoms near the CN, chunk", i+1, "of ", nChunk)
    
    return R_M[~MnearC]

def getSurfNN(R_M, R_M_all, Rnb = 3.0, chunkSize=100):
    nChunk = np.ceil(len(R_M)/chunkSize).astype(int)
    
    idxM = np.arange(len(R_M),dtype=int)
    Rsplit = np.array_split(R_M, nChunk)
    idxSpl = np.array_split(idxM, nChunk)
    isSurf = np.zeros(len(R_M),dtype=bool)
    
    for i in range(nChunk):
        numNb = np.sum(np.sum(((Rsplit[i])[:,np.newaxis,:] - R_M_all)**2,axis=2) < Rnb**2,axis=1)
        isSurf[(idxSpl[i])[numNb < 13]] = True
        print("Searching for surface sites using nearest neighbors, chunk", i+1, "of ", nChunk)
    
    return R_M[isSurf]

def getSurfVector(R_surfNN, R_M_all, Rnb = 15.0, angleCutoff = 30):
    idxSurf = np.zeros(len(R_surfNN), dtype=bool)
    
    for i in range(len(R_surfNN)):
        d = np.sqrt(np.sum((R_surfNN[i] - R_M_all)**2,axis=1))
        R_M_nb = R_M_all[(d<Rnb) & (d!=0)] - R_surfNN[i]
        R_M_nb = R_M_nb/(np.sqrt(np.sum(R_M_nb**2,axis=1)))[:,np.newaxis]    
        
        surfVec = -np.sum(R_M_nb, axis=0)
        surfVec = surfVec / np.linalg.norm(surfVec)
        
        angles = np.arccos(np.sum(surfVec * R_M_nb,axis=1))*180/np.pi
        
        idxSurf[i] = (np.sum(angles < angleCutoff) == 0)
        print("Search for surface sites using surface vectors, site", i+1, "of", len(R_surfNN))

    return R_surfNN[idxSurf]

def saveXYZ(R_list, Element_list, fileName):
    nAtoms = 0
    for R in R_list:
        nAtoms += len(R)
    with open(fileName, 'w') as file:
        file.write(str(nAtoms) + "\n")
        file.write("\n")
        for R, element in zip(R_list, Element_list):
            for i in range(len(R)):
                file.write(element + " " + str(R[i,0]) + " " + str(R[i,1]) + " " + str(R[i,2]) + "\n")

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

def getFeatPOSCAR(poscarDir, featParams):

    nPOSCAR = 0
    for file in os.listdir(poscarDir):
        nPOSCAR+=1

    featPOSCAR = np.zeros((nPOSCAR, featParams['nFeat']))

    for i,file in enumerate(os.listdir(poscarDir)):
        numA, coord = getCoord(poscarDir+"/"+file)
        featPOSCAR[i] = getFeat(numA, coord, featParams['n2b'], featParams['n3b'])

    return nPOSCAR, featPOSCAR

            
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

def loadNN(paramFile):
    nnParams = {}
    params = np.load(paramFile)
    nnParams['nL1'] = int(params['nL1'])
    nnParams['nL2'] = int(params['nL2'])
    nnParams['learningRate'] = float(params['learningRate'])
    nnParams['nEpoch'] = int(params['nEpoch'])
    return nnParams

def loadFeat(paramFile):
    featParams = {}
    params = np.load(paramFile)
    featParams['featA'] = params['featA']
    featParams['featB'] = params['featB']
    featParams['n2b'] = int(params['n2b'])
    featParams['n3b'] = int(params['n3b'])
    featParams['nFeat'] = int(params['nFeat'])
    return featParams
