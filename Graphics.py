# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 21:57:04 2019
Graphics Output
@author: rayanmosli
"""

import numpy as np
import matplotlib.pylab as plt
import os

class Graphics:
    def __init__(self,dataset):
        self.dataset=dataset
        self.class_name={}
        if 'CIFAR10' in dataset:
            self.class_name = {
                0: 'airplane',
                1: 'automobile',
                2: 'bird',
                3: 'cat',
                4: 'deer',
                5: 'dog',
                6: 'frog',
                7: 'horse',
                8: 'ship',
                9: 'truck',
                }
        elif 'MNIST' in dataset:
            self.class_name = {
                0: '0',
                1: '1',
                2: '2',
                3: '3',
                4: '4',
                5: '5',
                6: '6',
                7: '7',
                8: '8',
                9: '9',
                }
            
        elif 'imagenet' in dataset:
            synset_to_keras_idx = {}
            f = open(os.path.join('.','imagenet','synset_words.txt'),"r")
            idx = 0
            for line in f:
                parts = line.split(" ")
                synset_to_keras_idx[parts[0]] = idx
                self.class_name[idx] = " ".join(parts[1:])
                idx += 1
            f.close()
            
    def getClassName(self):
        return self.class_name
        
    def show(self,im,pred,save=False,path=None):
        c=self.getTitle(pred)
        if self.dataset=='CIFAR10':
            self.showCIFAR(im,c,save,path)
        elif self.dataset=='MNIST':
            self.showMNIST(im,c,save,path)
        elif 'imagenet224' in self.dataset:
            self.showImageNet224(im,c,save,path)
        elif 'imagenet299' in self.dataset:
            self.showImageNet299(im,c,save,path)
            
    def showCIFAR(self,im,c,save=False,path=None):
        if save==False:
            if np.min(im)<0:
                im=np.add(im,0.5)
            plt.imshow(im)
            plt.title("Class %s" % (self.getClassName()[c]))
            plt.axis('on')
            plt.show()
            plt.close()
        elif save==True:
            im=np.add(im,0.5)
            plt.imshow(im)
            plt.title("Class %s" % (self.getClassName()[c]))
            plt.axis('on')
            plt.savefig(path)
            plt.close()
            
    def showMNIST(self,im,c,save=False,path=None):
        if not im.shape == (28,28):
            im=im.reshape(28,28)
        if save==False:
            if np.min(im)<0:
                im=np.add(im,0.5)
            plt.imshow(im,cmap='gray')
            plt.title("Class %s" % (self.getClassName()[c]))
            plt.axis('on')
            plt.show()
            plt.close()
        elif save==True:
            im=np.add(im,0.5)
            plt.imshow(im,cmap='gray')
            plt.title("Class %s" % (self.getClassName()[c]))
            plt.axis('on')
            plt.savefig(path)
            plt.close()
    
    def showImageNet224(self,im,c,save=False,path=None):
        im=np.divide(im,2.0)
        im=np.add(im,0.5)
        if save==False:
            plt.figure()
            plt.imshow(im)
            plt.title("Class %s" %(c))
            plt.axis('on')
            plt.show()
            plt.close()
        elif save==True:
            plt.imshow(im)
            plt.title("Class %s" %(c))
            plt.axis('on')
            plt.savefig(path)
            plt.close()
            
    def showImageNet299(self,im,c,save=False,path=None):
        im=np.divide(im,2.0)
        im=np.add(im,0.5)
        if save==False:
            plt.figure()
            plt.imshow(im)
            plt.title("Class %s" %(c))
            plt.axis('on')
            plt.show()
            plt.close()
        elif save==True:
            plt.imshow(im)
            plt.title("Class %s" %(c))
            plt.axis('on')
            plt.savefig(path)
            plt.close()

    def getTitle(self,labels):
        if 'imagenet' in self.dataset:
            title="1: " + self.getClassName()[labels].strip()
        else:
            title=labels
        return title
