import py_util as pyu
import py_func as pyf
import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


nnParams, featParams = pyu.initParams()
nnParams['learningRate'] = 0.00001

nnParams["nEpoch"] = 25000

def getE_CuOCCOH(poscar):
    with open(poscar,'r') as p:
        return np.array(p.readline().split()[7], dtype=float)

batch1Dir = "CuOCCOH_batch1_results"

nCuOCCOH = 0
for file in os.listdir(batch1Dir):
    nCuOCCOH += 1
E_CuOCCOH = np.zeros(nCuOCCOH)

for i,file in enumerate(os.listdir(batch1Dir)):
    numA, coord = pyu.getCoord(batch1Dir + "/" + file)
    E_CuOCCOH[i] = getE_CuOCCOH(batch1Dir + "/" + file)

nCase, featCuOCCOH = pyu.getFeatPOSCAR(batch1Dir, featParams)

# df = pd.DataFrame(featCuOCCOH)
# df.to_csv("CuOCCOH_batch1_feat",header=False,index=False)
# pyf.trainE(featCuOCCOH, E_CuOCCOH, featParams, nnParams,save=True)

featParams=pyu.loadFeat("log/featParams.npz")
E_NN = pyf.getE(featCuOCCOH, featParams, nnParams)

plt.figure()
plt.hist(E_NN, alpha=0.3, label="NN")
plt.hist(E_CuOCCOH, alpha=0.3, label="DFT")
plt.legend()
plt.xlabel("E_CuOCCOH (eV)")