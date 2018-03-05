#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 11:38:06 2018

@author: hif1000
"""

import py_util as pyu
import py_func as pyf

import numpy as np

nnParams, featParams = pyu.initParams()

nAd, AdFeat, AdEngy = pyu.getAd("adsorbed", featParams)
nDe, DeFeat = pyu.getDe("desorbed", featParams)

'''
pyf.trainEL(AdFeat, AdEngy, DeFeat, featParams, nnParams,save=True)

#pyf.trainE(AdFeat, AdEngy, featParams, nnParams,save=True)

pyf.getE(AdFeat, featParams, nnParams)

pyf.getAd(AdFeat, featParams, nnParams)

'''

atoms, R = pyu.loadMCxyz("CuC_NP.xyz")

R_Cu = R[atoms==1]
R_C = R[atoms==0]

Cu_T = pyu.removeC(R_Cu, R_C, Rc=8.0, chunkSize=1000)

R_surfNN = pyu.getSurfNN(Cu_T, R_Cu, Rnb=3.0, chunkSize=100)

R_surf = pyu.getSurfVector(R_surfNN, R_Cu, Rnb=15.0, angleCutoff=30)

pyu.saveXYZ([R_surf], ['Cu'], "Cu_surf2.xyz")