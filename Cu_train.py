# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 21:04:37 2018

@author: hif10
"""

import py_util as pyu
import py_func as pyf

nnParams, featParams = pyu.initParams()

nnParams["nEpoch"] = 25000

nAd, AdFeat, AdEngy = pyu.getAd("adsorbed", featParams)
nDe, DeFeat = pyu.getDe("desorbed", featParams)

#pyf.trainEL(AdFeat, AdEngy, DeFeat, featParams, nnParams,save=True)

nnParams = pyu.loadNN("nnParams.npz")
featParams = pyu.loadFeat("featParams.npz")
E = pyf.getE(AdFeat, featParams, nnParams)
#
#pyf.getAd(AdFeat, featParams, nnParams)
