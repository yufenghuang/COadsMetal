#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 11:44:43 2018

@author: hif1000
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tf_func as tff

def scaleFeat(featParams, feat):
    return featParams['featA'] * feat + featParams['featB']

def getFeatAB(feat):
    featScaler = MinMaxScaler(feature_range=(0,1))
    featScaler.fit_transform(feat)
    nFeat = len(feat.T)
    featB = featScaler.transform(np.zeros((1,nFeat)))
    featA = featScaler.transform(np.ones((1,nFeat))) - featB
    return featA, featB

def loadNN(paramFile):
    nnParams = {}
    params = np.load(paramFile)
    nnParams['nL1'] = int(params['nL1'])
    nnParams['nL2'] = int(params['nL2'])
    nnParams['learningRate'] = float(params['learningRate'])
    nnParams['nEpoch'] = int(params['nEpoch'])
    return nnParams

def loadFeat(paramFile):
    featParams = {}
    params = np.load(paramFile)
    featParams['featA'] = params['featA']
    featParams['featB'] = params['featB']
    featParams['n2b'] = int(params['n2b'])
    featParams['n3b'] = int(params['n3b'])
    featParams['nFeat'] = int(params['nFeat'])
    return featParams

def trainEL(AdFeat, AdEngy, DeFeat, featParams, nnParams, save=False, load=False):        
    AdFeatTrain, AdFeatTest, AdEngyTrain, AdEngyTest = train_test_split(
            AdFeat, AdEngy, test_size=0.1)
    
    DeFeatTrain, DeFeatTest, DeLabelTrain, DeLabelTest = train_test_split(
            np.concatenate((AdFeat, DeFeat), axis=0), 
            np.concatenate((np.zeros_like(AdEngy), np.ones(len(DeFeat))), axis=0),
            test_size=0.1)
    
    tf_feat = tf.placeholder(tf.float32, (None,featParams['nFeat']))
    tf_engy = tf.placeholder(tf.float32, (None))
    tf_labels = tf.placeholder(tf.float32, (None))
    tf_LR = tf.placeholder(tf.float32)
    
    L3 = tff.getE(tf_feat, featParams['nFeat'], nnParams)
    L4 = tff.getAd(tf_feat, featParams['nFeat'], nnParams)
    
    engyLoss = tf.reduce_mean((L3 - tf_engy)**2)
    logitLoss = tf.reduce_mean((L4 - tf_labels)**2)
    
    with tf.variable_scope("optimizers", reuse=tf.AUTO_REUSE):
        optimizer3 = tf.train.AdamOptimizer(tf_LR).minimize(engyLoss)
        optimizer4 = tf.train.AdamOptimizer(tf_LR).minimize(logitLoss)
    
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        sess.run(tf.global_variables_initializer())
        
        if load:
            saver.restore(sess, "./model.ckpt")
        else:
            featParams['featA'], featParams['featB'] = getFeatAB(AdFeatTrain)
            
        AdTrainDict={tf_feat: scaleFeat(featParams, AdFeatTrain), tf_engy: AdEngyTrain, tf_LR: nnParams['learningRate']}
        AdTestDict ={tf_feat: scaleFeat(featParams, AdFeatTest),  tf_engy: AdEngyTest,  tf_LR: nnParams['learningRate']}
        
        DeTrainDict={tf_feat: scaleFeat(featParams, DeFeatTrain), tf_labels: DeLabelTrain, tf_LR: nnParams['learningRate']}
        DeTestDict ={tf_feat: scaleFeat(featParams, DeFeatTest),  tf_labels: DeLabelTest,  tf_LR: nnParams['learningRate']}
        
        for epoch in range(nnParams['nEpoch']):
            sess.run(optimizer3, feed_dict=AdTrainDict)
            sess.run(optimizer4, feed_dict=DeTrainDict)
            if epoch % 200 == 0:
                eLoss = sess.run(engyLoss, feed_dict=AdTrainDict)
                l4 = sess.run(L4, feed_dict=DeTrainDict)
                l4 = np.sum(np.array(l4 > 0.5, dtype=int) != DeLabelTrain)/len(DeLabelTrain)
                veLoss = sess.run(engyLoss, feed_dict=AdTestDict)
                vl4 = sess.run(L4, feed_dict=DeTestDict)
                vl4 = np.sum(np.array(vl4 > 0.5, dtype=int) != DeLabelTest)/len(DeLabelTest)
                print(epoch, eLoss, l4)
                print(epoch, veLoss, vl4)
                print(" ")
                
        eLoss = sess.run(engyLoss, feed_dict=AdTrainDict)
        l4 = sess.run(L4, feed_dict=DeTrainDict)
        l4 = np.sum(np.array(l4 > 0.5, dtype=int) != DeLabelTrain)/len(DeLabelTrain)
        veLoss = sess.run(engyLoss, feed_dict=AdTestDict)
        vl4 = sess.run(L4, feed_dict=DeTestDict)
        vl4 = np.sum(np.array(vl4 > 0.5, dtype=int) != DeLabelTest)/len(DeLabelTest)
        print(epoch, eLoss, l4)
        print(epoch, veLoss, vl4)
        print(" ")
        
        if save:
            np.savez("featParams", **featParams)
            np.savez("nnParams", **nnParams)
            savePath = saver.save(sess,"./model.ckpt")
            print("Model saved:", savePath)
        
def trainE(AdFeat, AdEngy, featParams, nnParams, save=False, load=False):        
    AdFeatTrain, AdFeatTest, AdEngyTrain, AdEngyTest = train_test_split(
            AdFeat, AdEngy, test_size=0.1)
        
    tf_feat = tf.placeholder(tf.float32, (None,featParams['nFeat']))
    tf_engy = tf.placeholder(tf.float32, (None))
#    tf_labels = tf.placeholder(tf.float32, (None))
    tf_LR = tf.placeholder(tf.float32)
    
    L3 = tff.getE(tf_feat, featParams['nFeat'], nnParams)
#    L4 = tff.getAd(tf_feat, featParams['nFeat'], nnParams, tf_labels)
    
    engyLoss = tf.reduce_mean((L3 - tf_engy)**2)
    
    with tf.variable_scope("optimizers", reuse=tf.AUTO_REUSE):
        optimizer3 = tf.train.AdamOptimizer(tf_LR).minimize(engyLoss)
    
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        sess.run(tf.global_variables_initializer())
        
        if load:
            saver.restore(sess, "./model.ckpt")
        else:
            featParams['featA'], featParams['featB'] = getFeatAB(AdFeatTrain)
            
        AdTrainDict={tf_feat: scaleFeat(featParams, AdFeatTrain), tf_engy: AdEngyTrain, tf_LR: nnParams['learningRate']}
        AdTestDict ={tf_feat: scaleFeat(featParams, AdFeatTest),  tf_engy: AdEngyTest,  tf_LR: nnParams['learningRate']}
            
        for epoch in range(nnParams['nEpoch']):
            sess.run(optimizer3, feed_dict=AdTrainDict)
            if epoch % 200 == 0:
                eLoss = sess.run(engyLoss, feed_dict=AdTrainDict)
                veLoss = sess.run(engyLoss, feed_dict=AdTestDict)
                print(epoch, eLoss)
                print(epoch, veLoss)
                print(" ")
    
        eLoss = sess.run(engyLoss, feed_dict=AdTrainDict)
        veLoss = sess.run(engyLoss, feed_dict=AdTestDict)
        print(epoch, eLoss)
        print(epoch, veLoss)
        print(" ")
        
        if save:
            np.savez("featParams", **featParams)
            np.savez("nnParams", **nnParams)
            savePath = saver.save(sess,"./model.ckpt")
            print("Model saved:", savePath)
            
def getE(feat, featParams, nnParams):
    tf_feat = tf.placeholder(tf.float32, (None,featParams['nFeat']))
    L3 = tff.getE(tf_feat, featParams['nFeat'], nnParams)
    
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./model.ckpt")
        
        feedDict = {tf_feat: scaleFeat(featParams, feat)}
        
        engy = sess.run(L3, feed_dict=feedDict)
        
    return engy

def getAd(feat, featParams, nnParams):
    tf_feat = tf.placeholder(tf.float32, (None,featParams['nFeat']))
    L4 = tff.getAd(tf_feat, featParams['nFeat'], nnParams)
    
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./model.ckpt")
        
        feedDict = {tf_feat: scaleFeat(featParams, feat)}
        
        label = np.array(sess.run(L4, feed_dict=feedDict) > 0.5, dtype=int)
        
    return label