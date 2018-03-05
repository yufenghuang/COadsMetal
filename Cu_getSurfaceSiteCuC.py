#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 11:38:06 2018

@author: hif1000
"""

import py_util as pyu

nnParams, featParams = pyu.initParams()

#atoms, R = pyu.loadMCxyz("CuC_NP.xyz")
#
#R_Cu = R[atoms==1]
#R_C = R[atoms==0]

atomList, atomType, R = pyu.loadXYZ("CuC_NP.xyz")

R_Cu = R[atomList == atomType.index('Cu')]
R_C = R[atomList == atomType.index('C')]

Cu_T = pyu.removeC(R_Cu, R_C, Rc=8.0, chunkSize=1000)

R_surfNN = pyu.getSurfNN(Cu_T, R_Cu, Rnb=3.0, chunkSize=100)

R_surf = pyu.getSurfVector(R_surfNN, R_Cu, Rnb=15.0, angleCutoff=30)

pyu.saveXYZ([R_surf], ['Cu'], "Cu_surf2.xyz")