# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 00:44:11 2018

@author: hif10
"""

import py_util as pyu
#import py_func as pyf

nnParams, featParams = pyu.initParams()

import os
import numpy as np

def getE(poscar):
    with open(poscar,'r') as p:
        E = np.array(p.readline().split(),dtype=float)
        return E[1]-E[0]-E[2]+0.5


def getCuOCCOH_E(poscar):
    with open(poscar, 'r') as p:
        print(poscar)
        line = p.readline().split()
        E_CuCO0 = float(line[1].strip("eV,"))
        E_CuCO1 = float(line[3].strip("V").strip("e"))
        E_CuOCCOH = float(line[5])
        E_Cu = float(line[8])
        E_CO = float(line[10])
        E_H2 = float(line[12])
    return E_CuCO0, E_CuCO1, E_CuOCCOH-E_Cu-2*E_CO-0.5*E_H2

AdDir = "CuOCCOH_results"
nAd = 0
for file in os.listdir(AdDir):
    nAd += 1

AdFeat = np.zeros((nAd, featParams['nFeat']))
AdEngy = np.zeros((nAd, 3))

iAd = 0
for file in os.listdir(AdDir):
    numA,coord=pyu.getCoord(AdDir+"/"+file)
    AdEngy[iAd] = np.array(getCuOCCOH_E(AdDir+"/"+file), dtype=float)
#    AdFeat[iAd] = pyu.getFeat(numA, coord, featParams['n2b'], featParams['n3b'])
    iAd += 1

#return nAd, AdFeat, AdEngy
