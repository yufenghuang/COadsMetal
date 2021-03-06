import py_util as pyu
import py_func as pyf
import numpy as np
import tensorflow as tf
import tf_func as tff
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

nnParams = pyu.loadNN("log/nnParams.npz")
featParams = pyu.loadFeat("log/featParams.npz")

tf_feat = tf.placeholder(tf.float32, (None,featParams['nFeat']))
L3 = tff.getE(tf_feat, featParams['nFeat'], nnParams)
Router = 8

atomList, atomType, R_surf = pyu.loadXYZ("surface_cu.xyz")
atomList, atomType, R_Cu = pyu.loadXYZ("cu_np.xyz")

Ei = np.zeros(len(R_surf))
Emask = np.zeros(len(R_surf))
with tf.Session() as sess:
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./log/model.ckpt")

    for i in range(len(R_surf)):
    # for Rs in R_surf[:10]:
        Rl = R_surf[i] - R_Cu
        dl = np.sqrt(np.sum(Rl**2, axis=-1))

        dl[dl>Router] = 0

        coord = np.zeros((np.sum(dl>0), 3))
        coord = Rl[dl>0]
        coord = np.concatenate((np.zeros((1,3)), coord), axis=0)

        feat = pyu.getFeat(len(coord), coord, featParams["n2b"], featParams["n3b"])

        feedDict = {tf_feat: pyf.scaleFeat(featParams, feat)}

        Ei[i] = sess.run(L3, feed_dict=feedDict)

        print("{}/{}".format(i, len(R_surf)), Ei[i])

    # pyu.saveXYZ([coord], ["Cu"], "test.xyz")

Ei = Ei-0.5

idx = np.zeros(len(Ei))
for k in range(3):
    idx[np.abs(R_surf[:,k]) < 2] = 1

with open("CuTao_surface.xyz", "w") as f:
    f.write("{}\n".format(len(Ei)))
    f.write(" \n")
    for i in range(len(Ei)):
        f.write("Cu {} {} {} {} \n".format(*R_surf[i], Ei[i]))


with open("CuTao_energies.xyz", "w") as f:
    f.write("{}\n".format(len(Ei)))
    f.write(" \n")
    for i in range(len(Ei)):
        f.write("Cu {} {} {} \n".format(Ei[i], Ei[i], Ei[i]))


for i in range(len(Ei)):
    if R_surf[i, 0] > np.mean(R_surf[:,0]) and (np.abs(R_surf[i,2]-np.mean(R_surf[:,2])) < 100):
        print(R_surf[i, 0], np.mean(R_surf[:, 0]), R_surf[i,2], np.mean(R_surf[:,2]))
        Emask[i] = 1

plt.figure()
plt.title("Nanoparticle")
plt.hist(Ei, alpha=0.5, bins=15)

def plt_hist(axis, data, hatch, label, bins=None):
    if bins is None:
        bins = int(len(data) ** .5)
    counts, edges = np.histogram(data, bins=bins)
    edges = np.repeat(edges, 2)
    hist = np.hstack((0, np.repeat(counts, 2), 0))

    outline, = axis.plot(edges,hist,linewidth=1.3)
    axis.fill_between(edges,hist,0,
                edgecolor=outline.get_color(), hatch = hatch, label=label,
                facecolor = 'none')  ## < removes facecolor
    axis.set_ylim(0, None, auto = True)

plt.figure()
plt.title("CO Adsorption Energies on a Copper Nanoparticle \n", fontsize=16)
# plt.hist(Ei[Emask > 0], alpha=0.5, bins=15)
plt_hist(plt.gca(), Ei[Emask>0], '/////', label="Rand1", bins=10)
plt.ylabel("Count", fontsize=16)
plt.xlabel("$E_{CO}$ (eV)", fontsize=16)
plt.ylim(ymin=0, ymax=1800)
plt.vlines(-1.07, 0, 1800, linestyles='--')
plt.vlines(-0.87, 0, 1800, linestyles='--')
plt.vlines(-0.78, 0, 1800, linestyles='--')
plt.text(-1.2, 1820, "Cu(211)", fontsize=14)
plt.text(-1.0, 1820, "Cu(100)", fontsize=14)
plt.text(-0.8, 1820, "Cu(111)", fontsize=14)
plt.xticks(fontsize=16)
plt.yticks([0, 300, 600, 900, 1200, 1500, 1800], fontsize=16)
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.savefig("COads_TaoNP.pdf")



Ei2 = Ei[Emask>0]
Rs2 = R_surf[Emask>0]
with open("CuTao_surface2.xyz", "w") as f:
    f.write("{}\n".format(len(Ei2)))
    f.write(" \n")
    for i in range(len(Ei2)):
        f.write("Cu {} {} {} {} \n".format(*Rs2[i], Ei2[i]))
with open("CuTao_energies2.xyz", "w") as f:
    f.write("{}\n".format(len(Ei2)))
    f.write(" \n")
    for i in range(len(Ei2)):
        f.write("Cu {} {} {} \n".format(Ei2[i], Ei2[i], Ei2[i]))



# for i in range(len(Ei)):
#     if Ei[i] <= -0.9:
#         Rl = R_surf[i] - R_Cu
#         dl = np.sqrt(np.sum(Rl**2, axis=-1))
#         dl[dl>Router] = 0
#         coord = np.zeros((np.sum(dl>0), 3))
#         coord = Rl[dl>0]
#         coord = np.concatenate((np.zeros((1,3)), coord), axis=0)
#         pyu.saveXYZ([coord], ["Cu"], "COcluster/surface_site"+str(i)+".xyz", comment="E_CO: " + str(Ei[i]))
