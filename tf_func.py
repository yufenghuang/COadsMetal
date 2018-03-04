#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 11:13:05 2018

@author: hif1000
"""

import tensorflow as tf

def getNN(nnIn, nIn, nOut, layer):
    with tf.variable_scope(layer,reuse=tf.AUTO_REUSE):
        W = tf.get_variable("weights", shape=[nIn, nOut], \
                            initializer=tf.contrib.layers.xavier_initializer())
        B = tf.get_variable("biases", shape=[nOut], \
                            initializer=tf.zeros_initializer())
        nnOut = tf.matmul(nnIn, W) + B
    return nnOut
    
def getE(tf_feat, nFeat, nnParams):
            
    L1 = tf.nn.sigmoid(getNN(tf_feat, nFeat, nnParams['nL1'], 'layer1'))
    L2 = tf.nn.sigmoid(getNN(L1, nnParams['nL1'], nnParams['nL2'], 'layer2'))
    L3 = tf.reshape(getNN(L2, nnParams['nL2'], 1, 'layer3'),[-1])
        
    return L3

def getAd(tf_feat, nFeat, nnParams):
    with tf.variable_scope("adsorption", reuse=tf.AUTO_REUSE):
        mu = tf.get_variable("mu", shape=[1],initializer=tf.constant_initializer(-0.3))
    L3 = getE(tf_feat, nFeat, nnParams)
    L4 = tf.tanh((L3-mu)/0.0257)*0.5+0.5
    return L4

def getW(layer):
    with tf.variable_scope(layer,reuse=tf.AUTO_REUSE):
        W = tf.get_variable("weights")
    return(W)
    
def getB(layer):
    with tf.variable_scope(layer,reuse=tf.AUTO_REUSE):
        B = tf.get_variable("biases")
    return B

def getMu():
    with tf.variable_scope("adsorption", reuse=tf.AUTO_REUSE):
        mu = tf.get_variable("mu")
    return mu