# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 21:33:33 2019
Dataset
@author: rayanmosli
"""

from keras.models import model_from_json
import numpy as np
import os
from keras.applications.inception_v3 import InceptionV3
from keras.utils import to_categorical
from setup_cifar import CIFAR
from setup_mnist import MNIST
from setup_inception import ImageNet
from Train_Transfer_Models.CIFAR10.CNN_Capsule import Capsule
import keras


def Dataset(dataset,pars=False,targeted=False):
    if dataset=='CIFAR10':
        testData,testLabels=loadCIFARData()
        return testData,testLabels
    elif dataset=='MNIST':
        testData,testLabels=loadMNISTData()
        return testData,testLabels
    elif dataset=='imagenet224':
        testData,testLabels=loadImageNet224Data()
        return testData,testLabels
    elif dataset=='imagenet299':
        testData,testLabels=loadImageNet299Data(pars,targeted)
        return testData,testLabels

def loadMNISTData():
    data=MNIST()
    testData=[]
    testLabels=[]
    for d,l in zip(data.test_data,data.test_labels):
        testData.append(d)
        testLabels.append(l)
    return testData,testLabels

def loadMNISTModel():
    json_file = open(os.path.join("models","mnist.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join("models","mnist.h5"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
        
        
def loadMNISTHRNNModel():
    json_file = open(os.path.join(".",'Train_Transfer_Models','MNIST','models',"HRNN.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(".",'Train_Transfer_Models','MNIST','models',"HRNN.h5"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
def loadMNISTMLPModel():
    json_file = open(os.path.join(".",'Train_Transfer_Models','MNIST','models',"MLP.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(".",'Train_Transfer_Models','MNIST','models',"MLP.h5"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
def loadMNISTCNNModel():
    json_file = open(os.path.join(".",'Train_Transfer_Models','MNIST','models',"CNN.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(".",'Train_Transfer_Models','MNIST','models',"CNN.h5"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def loadMNISTSIAMESEMLPModel():
    json_file = open(os.path.join(".",'Train_Transfer_Models','MNIST','models',"Siamese_MLP.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(".",'Train_Transfer_Models','MNIST','models',"Siamese_MLP.h5"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
    
def loadCIFAR10RESNETModel():
    json_file = open(os.path.join(".",'Train_Transfer_Models','CIFAR10','models',"ResNet.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(".",'Train_Transfer_Models','CIFAR10','models',"ResNet.h5"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def loadCIFAR10CNNCapsuleNoAugModel():
    json_file = open(os.path.join(".",'Train_Transfer_Models','CIFAR10','models',"CNN_Capsule_NoAug.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json,custom_objects={'Capsule': Capsule(10,16)})
    model.load_weights(os.path.join(".",'Train_Transfer_Models','CIFAR10','models',"CNN_Capsule_NoAug.h5"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model    

def loadCIFAR10CNNCapsuleAugModel():
    json_file = open(os.path.join(".",'Train_Transfer_Models','CIFAR10','models',"CNN_Capsule_WithAug.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json,custom_objects={'Capsule': Capsule(10,16)})
    model.load_weights(os.path.join(".",'Train_Transfer_Models','CIFAR10','models',"CNN_Capsule_WithAug.h5"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model    

def loadCIFAR10simpleDeepCNNNoAugModel():
    json_file = open(os.path.join(".",'Train_Transfer_Models','CIFAR10','models',"simpleDeepCNN_NoAug.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(".",'Train_Transfer_Models','CIFAR10','models',"simpleDeepCNN_NoAug.h5"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model   

def loadCIFAR10simpleDeepCNNWithAugModel():
    json_file = open(os.path.join(".",'Train_Transfer_Models','CIFAR10','models',"simpleDeepCNN_WithAug.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(".",'Train_Transfer_Models','CIFAR10','models',"simpleDeepCNN_WithAug.h5"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model   
    
def loadCIFARData():
    data=CIFAR()
    testData=[]
    testLabels=[]
    for d,l in zip(data.test_data,data.test_labels):
        testData.append(d)
        testLabels.append(l)
    return testData,testLabels

def loadCIFARModel():
    json_file = open(os.path.join("models","cifar.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join("models","cifar.h5"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
def loadImageNet224Model():
    model=InceptionV3(include_top=True,weights='imagenet',input_tensor=keras.Input(shape=(224,224,3)),pooling=None,classes=1000)
    return model
    
def loadImageNet224Data():
    testData=np.load(os.path.join('.','imagenet','x_val_0_10000.npy'))[:1000]
    testLabels=np.load(os.path.join('.','imagenet','y_val.npy'))[:1000]
    testLabels = to_categorical(testLabels, 1000)
    return testData,testLabels
    
def loadImageNet299Model():
    model=InceptionV3(include_top=True,weights='imagenet',classes=1000)
    return model
    
def loadImageNet299Data(pars=False,targeted=False):
    if pars:
        if not targeted:
            indices=np.load('indices_untargeted.npy')
            imagenet=ImageNet(indices=indices)
        elif targeted:
            indices=np.load('indices_targeted.npy')
            imagenet=ImageNet(indices=indices)
    else:
        imagenet=ImageNet(indices=[])
    testData=imagenet.test_data
    testLabels=imagenet.test_labels
    return testData,testLabels
    
def returnDimensions(dataset):
    if dataset=='imagenet224':
        return 224,224,3
    elif dataset=='imagenet299':
        return 299,299,3
    elif dataset=='MNIST':
        return 28,28,1
    elif dataset=='CIFAR10':
        return 32,32,3
    else:
        return None,None,None