#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 11:38:06 2018

@author: hif1000
"""

import py_util as pyu
import py_func as pyf

nnParams, featParams = pyu.initParams()

nAd, AdFeat, AdEngy = pyu.getAd("adsorbed", featParams)
nDe, DeFeat = pyu.getDe("desorbed", featParams)

pyf.trainEL(AdFeat, AdEngy, DeFeat, featParams, nnParams,save=True)

#pyf.trainE(AdFeat, AdEngy, featParams, nnParams,save=True)

pyf.getE(AdFeat, featParams, nnParams)

pyf.getAd(AdFeat, featParams, nnParams)