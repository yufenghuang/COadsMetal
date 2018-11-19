# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 21:04:37 2018

@author: hif10
"""

import py_util as pyu
import py_func as pyf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

nnParams, featParams = pyu.initParams()

nnParams["nEpoch"] = 10000

nnParams['nL1'] = 40
nnParams['nL2'] = 40

nAd, AdFeat, AdEngy = pyu.getAd("adsorbed", featParams)
nDe, DeFeat = pyu.getDe("desorbed", featParams)

AdFeatTrain, AdFeatTestValid, AdEngyTrain, AdEngyTestValid = train_test_split(
    AdFeat, AdEngy, test_size=0.2)

DeFeatTrain, DeFeatTestValid, DeLabelTrain, DeLabelTestValid = train_test_split(
    np.concatenate((AdFeat, DeFeat), axis=0),
    np.concatenate((np.zeros_like(AdEngy), np.ones(len(DeFeat))), axis=0),
    test_size=0.2)

AdFeatValid, AdFeatTest, AdEngyValid, AdEngyTest = train_test_split(
    AdFeatTestValid, AdEngyTestValid, test_size=0.5)

DeFeatValid, DeFeatTest, DeLabelValid, DeLabelTest = train_test_split(
    DeFeatTestValid, DeLabelTestValid, test_size=0.5)

featSets = [AdFeatTrain, AdFeatValid, AdFeatTest, DeFeatTrain, DeFeatValid, DeFeatTest]
engySets = [AdEngyTrain, AdEngyValid, AdEngyTest, DeLabelTrain, DeLabelValid, DeLabelTest]

'''
featSets = np.load("features.npz")
engySets = np.load("energies.npz")
featSets = [featSets[key] for key in featSets.files][0]
engySets = [engySets[key] for key in engySets.files][0]
'''

RMSE_train, RMSE_valid, mu, w, b = pyf.trainEL_validation(featSets, engySets, featParams, nnParams, load=False, save=False)

i = 5
plt.figure()
plt.plot(RMSE_valid[i:])
plt.plot(RMSE_train[i:])


'''
i = 30
nnParams["nEpoch"] = 170 * 20
RMSE_train2, RMSE_valid2, mu, w, b = pyf.trainEL_validation(featSets, engySets, featParams, nnParams, load=True, save=False)
plt.figure()
plt.plot(RMSE_valid2[i:])
plt.plot(RMSE_train2[i:])
plt.plot(RMSE_valid[i:len(RMSE_valid2)], 'o')
plt.plot(RMSE_train[i:len(RMSE_train2)], 's')
'''

'''
i = 30
nnParams["nEpoch"] = 170 * 20
RMSE_train2, RMSE_valid2, mu, w, b = pyf.trainEL_validation(featSets, engySets, featParams, nnParams, load=True, save=True)
plt.figure()
# plt.plot(RMSE_valid2[i:])
# plt.plot(RMSE_train2[i:])
plt.plot(RMSE_valid[i:len(RMSE_valid2)*2])
plt.plot(RMSE_train[i:len(RMSE_train2)*2])

np.savez("RMSE", RMSE_valid=RMSE_valid, RMSE_train=RMSE_train)
np.savez("features", featSets)
np.savez("energies", engySets)
'''

# pyf.trainEL(AdFeat, AdEngy, DeFeat, featParams, nnParams,save=True)

# nnParams = pyu.loadNN("log/nnParams.npz")
# featParams = pyu.loadFeat("log/featParams.npz")
# E = pyf.getE(AdFeat, featParams, nnParams)
#
# pyf.getAd(AdFeat, featParams, nnParams)
