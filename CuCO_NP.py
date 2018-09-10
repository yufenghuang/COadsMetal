import py_util as pyu
import py_func as pyf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

nnParams = pyu.loadNN("log/nnParams.npz")
featParams = pyu.loadFeat("log/featParams.npz")

# nnParams["nEpoch"] = 20000

atomList, atomType, R = pyu.loadXYZ('Cu_NP_n099_dump.xyz')
R_Cu = R[atomList == atomType.index('Cu')]

# atomList, atomType, R = pyu.loadXYZ("CuC_NP.xyz")

# R_Cu = R[atomList == atomType.index('Cu')]
# R_C = R[atomList == atomType.index('C')]

# Cu_T = pyu.removeC(R_Cu, R_C, Rc=8.0, chunkSize=1000)

R_surfNN = pyu.getSurfNN(R_Cu, R_Cu, Rnb=3.0, chunkSize=100)

R_surf = pyu.getSurfVector(R_surfNN, R_Cu, Rnb=15.0, angleCutoff=30)

for Rs in np.array_split(R_surf, 10):
    print(len(Rs))
