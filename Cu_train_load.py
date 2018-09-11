import py_util as pyu
import py_func as pyf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

nnParams = pyu.loadNN("log/nnParams.npz")
featParams = pyu.loadFeat("log/featParams.npz")

# nnParams["nEpoch"] = 20000

featSets = np.load("features.npz")
engySets = np.load("energies.npz")
featSets = [featSets[key] for key in featSets.files][0]
engySets = [engySets[key] for key in engySets.files][0]

RMSEs = np.load("RMSE.npz")
RMSE_train = RMSEs["RMSE_train"]
RMSE_valid = RMSEs["RMSE_valid"]

plt.figure()
start = 0
stop = 375
epochs = np.arange(start, stop)*20
plt.plot(epochs, np.sqrt(RMSE_train[start:stop]))
plt.plot(epochs, np.sqrt(RMSE_valid[start:stop]))
plt.title("Model Training")
plt.ylabel("Root Mean Squared Error (eV)")
plt.xlabel("Training iteration")
plt.legend(["Training Set", "Validation Set"])
# plt.arrow(5000, 0.2, 1500, 0.06, head_width=0.01)
ax = plt.gca()
ax.annotate("Training is terminated at \n the 5000th iteration to \n avoid over-training",
            xy=(5000, 0.12), xytext=(4800, 0.16),arrowprops=dict(arrowstyle="->"))
plt.savefig("model_training.pdf")

pyf.trainEL_getError(featSets, engySets, featParams, nnParams)