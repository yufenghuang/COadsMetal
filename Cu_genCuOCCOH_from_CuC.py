# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 21:11:07 2018

@author: hif10
"""

import py_util as pyu
import py_func as pyf
import numpy as np

nnParams = pyu.loadNN("nnParams.npz")
featParams = pyu.loadFeat("featParams.npz")

atomList, atomType, R_surf = pyu.loadXYZ('Cu_surf2.xyz')
atomList, atomType, R = pyu.loadXYZ('CuC_NP.xyz')
R_Cu = R[atomList == atomType.index('Cu')]

nCase = 500
Rcut = 20.0 # radius for the initial cluster
Rcut4 = 15.0
Rcut2 = 3.0 # nearest neighbor distance
Rcut3 = 8.0 # 
angleCutoff=45

dCu0C0 = [2.0, 2.56, 2.59, 2.0, 1.98]
dCu1C1 = [2.0, 2.0, 2.0, 2.14, 2.19]
dC0C1 = [1.47, 1.41, 1.34, 1.45,1.36]
dC0O0 = [1.39, 1.41, 1.41, 1.39, 1.4]
dC1O1 = [1.22, 1.19, 1.19, 1.24, 1.2]
dC0H = [1.91, 1.91, 1.91, 1.91, 1.92]
aOCH = [30, 30, 30, 29.2, 29.4]

dCu0C0 = np.mean(dCu0C0)
dCu1C1 = np.mean(dCu1C1)
dC0C1 = np.mean(dC0C1)
dC0O0 = np.mean(dC0O0)
dC1O1 = np.mean(dC1O1)
dC0H = np.mean(dC0H)
aOCH = np.mean(aOCH)*np.pi/180

randomIdx = np.random.permutation(len(R_surf))[:nCase]

Rs = R_surf[randomIdx]

for iCase in range(nCase):
    print(iCase)
    d_sq = np.sum((Rs[iCase] - R_Cu)**2,axis=1)
    Rcluster = R_Cu[d_sq < Rcut**2]
    RNN = R_Cu[(d_sq < Rcut2**2) & (d_sq != 0)]
    Rcluster2 = np.concatenate([(Rs[iCase])[np.newaxis,:],R_Cu[(d_sq < Rcut3**2) & (d_sq !=0)]],axis=0)
    
    feat=pyu.getFeat(len(Rcluster2), Rcluster2-Rs[iCase],featParams['n2b'],featParams['n3b'])
    E0 = pyf.getE(feat[np.newaxis,:], featParams, nnParams)[0]
    print(E0)
    if E0 < -0.5:
        R_surfNN = pyu.getSurfNN(RNN, Rcluster, Rnb=Rcut2, chunkSize=100)
        R_surf = pyu.getSurfVector(R_surfNN, Rcluster, Rnb=Rcut4, angleCutoff=angleCutoff)
            
        feat = np.zeros([len(R_surf), featParams['nFeat']])
        for iNN in range(len(R_surf)):
    #        print(R_surf[iNN] - Rs[iCase])
            diNN_sq = np.sum((R_surf[iNN] - Rcluster)**2,axis=1)
            RiNN = np.zeros((np.sum(diNN_sq < Rcut3**2), 3))
            RiNN[1:] = Rcluster[(diNN_sq != 0.0 ) & (diNN_sq < Rcut3**2)]-R_surf[iNN]
    #        pyu.saveXYZ([RiNN], ['Cu'], 'CuOCCOH_poscar/Cu'+str(iNN)+'.xyz')
            feat[iNN] = pyu.getFeat(len(RiNN), RiNN,featParams['n2b'],featParams['n3b'])
        
        RCu0 = Rs[iCase]
        E = pyf.getE(feat, featParams, nnParams)
        if len(E) != 0:
            E1 = np.min(E)
            RCu1 = R_surf[np.argmin(E)]
        
            vector1 = -np.sum(Rcluster - RCu0,axis=0)
            vector2 = RCu1 - RCu0
            
            vector1 = np.cross(np.cross(vector2, vector1),vector2)
            vector1 = vector1/np.linalg.norm(vector1)
            vector2 = vector2/np.linalg.norm(vector2)
                    
            dCuCu = np.linalg.norm(RCu1 - RCu0)
            
            dCuC1 = dCu0C0 * dCuCu / (dCuCu - dC0C1)
            dCuC2 = dCu1C1 * dCuCu / (dCuCu - dC0C1)
            
            aCu0 = np.arccos((dCuCu**2 + dCuC1**2 - dCuC2**2)/(2*dCuCu*dCuC1))
            aCu1 = np.arccos((dCuCu**2 + dCuC2**2 - dCuC1**2)/(2*dCuCu*dCuC2))
            
            RC0 = np.cos(aCu0)*dCu0C0*vector2 + np.sin(aCu0)*dCu0C0*vector1 + RCu0
            RC1 = -np.cos(aCu1)*dCu1C1*vector2 + np.sin(aCu1)*dCu1C1*vector1 + RCu1
            
            aC0 = (2*np.pi - (np.pi-aCu0))/2
            aC1 = (2*np.pi - (np.pi-aCu1))/2
            
            RO0 = np.cos(aC0)*dC0O0*vector2 + np.sin(aC0)*dC0O0*vector1 + RC0
            RO1 = -np.cos(aC1)*dC1O1*vector2 + np.sin(aC1)*dC1O1*vector1 + RC1
            
            aH = aC0 - aOCH
            RH = np.cos(aH)*dC0H*vector2 + np.sin(aH)*dC0H*vector1+RC0
            
        #    pyu.saveXYZ([np.array([RC0, RC1]), Rcluster2], ['C','Cu'], 'CuOCCOH_poscar/Cu'+str(iCase)+'.xyz')
            
            Rout = np.zeros((5+len(Rcluster2),3))
            Rout[5:] = Rcluster2
            Rout[0] = RH
            Rout[1] = RO0
            Rout[2] = RO1
            Rout[3] = RC0
            Rout[4] = RC1
            Rout = Rout - RCu0
            with open('CuOCCOH_poscar/Cu'+str(iCase).zfill(3)+'POSCAR_CuOCCOH', 'w') as pFile:
                pFile.write("Cu0-CO: " + str(E0) + "eV, Cu1-CO: " + str(E1) +"eV \n")
                pFile.write("1.0\n")
                pFile.write("20.0 0.0 0.0\n")
                pFile.write("0.0 20.0 0.0\n")
                pFile.write("0.0 0.0 20.0\n")
                pFile.write("H O C Cu\n")
                pFile.write("1 2 2 " + str(len(Rcluster2))+"\n")
                pFile.write("Selective Dynamics\n")
                pFile.write("Cartesian\n")
                for i in range(5):
                    pFile.write(str(Rout[i,0]) + " " + str(Rout[i,1]) + " " + str(Rout[i,2]) + " T T T \n")
                for i in range(5, len(Rout)):
                    pFile.write(str(Rout[i,0]) + " " + str(Rout[i,1]) + " " + str(Rout[i,2]) + " F F F \n")
                    
            Rout = Rcluster2
            with open('CuOCCOH_poscar/Cu'+str(iCase).zfill(3)+'POSCAR', 'w') as pFile:
                pFile.write("Cu0-CO: " + str(E0) + "eV, Cu1-CO: " + str(E1) +"eV \n")
                pFile.write("1.0\n")
                pFile.write("20.0 0.0 0.0\n")
                pFile.write("0.0 20.0 0.0\n")
                pFile.write("0.0 0.0 20.0\n")
                pFile.write("Cu\n")
                pFile.write(str(len(Rcluster2))+"\n")
                pFile.write("Cartesian\n")
                for i in range(len(Rout)):
                    pFile.write(str(Rout[i,0]) + " " + str(Rout[i,1]) + " " + str(Rout[i,2]) + "\n")
