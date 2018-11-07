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

plt.figure()
plt.title("Test Set", fontsize=14)
Ep = pyf.getE(featSets[2], featParams, nnParams)-0.5
plt.hist(engySets[2]-0.5, range=(-1.4, -0.5), alpha=0.5)
plt.hist(Ep, alpha=0.5, range=(-1.4, -0.5))
plt.xlabel("CO adsorption energy (eV)", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.legend(["DFT", "Neural Network"], fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("test_hist.pdf")
print("Test set (DFT): [{}, {}]".format(np.min(engySets[2])-0.5, np.max(engySets[2])-0.5))
print("Test set (NN): [{}, {}]".format(np.min(Ep), np.max(Ep)))


plt.figure()
plt.title("Validation Set", fontsize=14)
Ep = pyf.getE(featSets[1], featParams, nnParams)-0.5
plt.hist(engySets[1]-0.5, range=(-1.4, -0.5), alpha=0.5)
plt.hist(Ep, alpha=0.5, range=(-1.4, -0.5))
plt.xlabel("CO adsorption energy (eV)", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.legend(["DFT", "Neural Network"], fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("valid_hist.pdf")
print("Validation set (DFT): [{}, {}]".format(np.min(engySets[1])-0.5, np.max(engySets[1])-0.5))
print("Validation set (NN): [{}, {}]".format(np.min(Ep), np.max(Ep)))


plt.figure()
plt.title("Training Set", fontsize=14)
Ep = pyf.getE(featSets[0], featParams, nnParams)-0.5
plt.hist(engySets[0]-0.5, range=(-1.4, -0.5), alpha=0.5)
plt.hist(Ep, alpha=0.5, range=(-1.4, -0.5))
plt.xlabel("CO adsorption energy (eV)", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.legend(["DFT", "Neural Network"], fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("train_hist.pdf")
print("Training set (DFT): [{}, {}]".format(np.min(engySets[0])-0.5, np.max(engySets[0])-0.5))
print("Training set (NN): [{}, {}]".format(np.min(Ep), np.max(Ep)))


pyf.trainEL_getError(featSets, engySets, featParams, nnParams)