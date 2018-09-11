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


atomList, atomType, R = pyu.loadXYZ('Cu_NP_n096_dump.xyz')
R_Cu = R[atomList == atomType.index('Cu')]
R_surfNN = pyu.getSurfNN(R_Cu, R_Cu, Rnb=3.0, chunkSize=100)
R_surf = pyu.getSurfVector(R_surfNN, R_Cu, Rnb=15.0, angleCutoff=30)

Ei = np.zeros(len(R_surf))
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

        print(i, Ei[i])

    # pyu.saveXYZ([coord], ["Cu"], "test.xyz")

idx = np.zeros(len(Ei))
for k in range(3):
    idx[np.abs(R_surf[:,k]) < 2] = 1

plt.figure()
plt.title("Grain Boundary")
plt.hist(Ei[idx>0], alpha=0.5)
plt.figure()
plt.title("Surface")
plt.hist(Ei[idx<1], alpha=0.5)

with open("Cu_NP_n096_surf.xyz", "w") as f:
    f.write("{}\n".format(len(Ei)))
    f.write(" \n")
    for i in range(len(Ei)):
        print(i)
        f.write("Cu {} {} {} {} \n".format(*R_surf[i], Ei[i]*10))

with open("Cu_NP_n096_engy.xyz", "w") as f:
    f.write("{}\n".format(len(Ei)))
    f.write(" \n")
    for i in range(len(Ei)):
        print(i)
        f.write("Cu  {} {} {} \n".format(Ei[i]*10,  Ei[i]*10, Ei[i]*10))
