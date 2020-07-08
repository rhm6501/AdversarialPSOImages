## setup_cifar.py -- cifar data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import numpy as np
import os
import urllib.request


class CIFAR:
    def __init__(self):
        train_data = []
        train_labels = []
        
        if not os.path.exists("cifar-10-batches-bin"):
            urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                                       "cifar-data.tar.gz")
            os.popen("tar -xzf cifar-data.tar.gz").read()
            

        for i in range(5):
            r,s = self.load_batch(os.path.join("cifar-10-batches-bin","data_batch_"+str(i+1)+".bin"))
            train_data.extend(r)
            train_labels.extend(s)
            
        train_data = np.array(train_data,dtype=np.float32)
        train_labels = np.array(train_labels)
        
        self.test_data, self.test_labels = self.load_batch(os.path.join("cifar-10-batches-bin", "test_batch.bin"))
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]      
    def load_batch(self,fpath):
        f = open(fpath,"rb").read()
        size = 32*32*3+1
        labels = []
        images = []
        for i in range(10000):
            arr = np.fromstring(f[i*size:(i+1)*size],dtype=np.uint8)
            lab = np.identity(10)[arr[0]]
            img = arr[1:].reshape((3,32,32)).transpose((1,2,0))
    
            labels.append(lab)
            images.append((img/255)-.5)
        return np.array(images),np.array(labels)
        