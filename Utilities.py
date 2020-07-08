# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 21:33:55 2019
Utilities
@author: rayanmosli
"""
import numpy as np
from math import exp

def calculateFitnessPenalty(diff,penaltyWeight):
    penalty=(diff*penaltyWeight)
    return penalty

def compareImages(image1,image2,dLength):
    diff=np.sum(np.square(np.subtract(image1,image2)))/dLength  
    return diff

def compareImagesL0(image1,image2):
    diff=np.sum([0 if image1[i] == image2[i] else 1 for i,_ in enumerate(image1)]) 
    return diff
    
def compareImagesLinfty(image1,image2):
    diff=np.max(np.abs(np.subtract(image1,image2)))
    return diff
    
def SSIM(image1,image2):
    image1=np.add(image1,0.5)
    image2=np.add(image2,0.5)
    m1=np.mean(image1)
    m2=np.mean(image2)
    var1=np.var(image1)
    var2=np.var(image2)
    cov=np.mean(np.multiply(image1,image2))-(m1*m2)
    L=255
    c1=(0.01*L)**2
    c2=(0.03*L)**2
    ssim=(((2*m1*m2)+c1)*((2*cov)+c2))/(((m1**2)+(m2**2)+c1)*((var1**2)+(var2**2)+c2))
    ssim=(1-ssim)/2
    return ssim

def findImageStructuralInformation(varVector):
    sortedVarList=sorted(varVector)
    lowVarThreshold=np.percentile(sortedVarList,25)
    highVarIndices=[]
    lowVarIndices=[]
    for i,x in enumerate(varVector):
        if x>lowVarThreshold:
            highVarIndices.append(i)
        else:
            lowVarIndices.append(i)
    return lowVarIndices,highVarIndices

def calculateModelAcc(model,data,labels):
    r=[]
    for x,y in zip(data,labels):
        pred=predictSample(model,x)
        r.append(np.argmax(pred) == np.argmax(y))
    print(np.mean(r))

def flattenImage(sample):
    img = sample.flatten()
    return img


def reshapeImage(sample,dataset):
    newSample=None
    if 'CIFAR10' in dataset:
        newSample=np.reshape(sample,(32,32,3))
    elif 'MNIST' in dataset:
        newSample=np.reshape(sample,(28,28,1))
    elif 'imagenet224' in dataset:
        newSample=np.reshape(sample,(224,224,3))
    elif 'imagenet299' in dataset:
        newSample=np.reshape(sample,(299,299,3))
    return newSample

def predictSample(model,sample):
    pred = model.predict(np.array([sample])).astype(np.float64)
    if not np.round(np.sum(pred[0]),decimals=2)==1.0 or (np.round(np.sum(pred[0]),decimals=2)==1.0 and any(n < 0 for n in pred[0])):
        pred[0]=softmax(pred[0])
    return pred
   
def calculateDiffAndPredict(image1,image2,model,dataset):
    diff=compareImages(image1,image2,len(image2))
    z=reshapeImage(image2,dataset)
    predAfter=predictSample(model,z)
    return diff,z,predAfter
     
def nLargestValues(lst,stepSize):
    n=sorted(range(len(lst)), key=lambda i: lst[i])[-1*stepSize:]
    return n
        
def common_elements(list1, list2):
    return [element for element in list1 if element in list2]
    
def common_nonzero_elements(self,list1, list2):
    return [element for element in list1 if element in list2 and element==0]
                    
def list_rindex(li, x):
    for i in reversed(range(len(li))):
        if li[i] == x:
            return i
    raise ValueError("{} is not in list".format(x))
  
def chunks(l, n):
    for i in range(0, len(l), n):
        if len(l)-n < i+n :
            yield l[i:]
        else:
            yield l[i:i+n]
            
def groupIndices(lst,rowSize,channels):
    l=sorted(lst)
    chunks = [[l[0]]]
    for value in l[1:]:
        diff=[]
        for v in chunks[-1]:
            diff.append((abs(value%(rowSize*channels)-v%(rowSize*channels)),abs(np.floor(value/(rowSize*channels))-np.floor(v/(rowSize*channels)))))
        if np.any([True for d,r in diff if d==1 or d==0 and r<=1]):
            chunks[-1].append(value)
        else:
            chunks.append([value])
    return chunks

def separateToNeighborhoods(indices,height,width,channels,numberOfNeighborhoods):
    tiles=splitImageIntoNRegions(height,width,channels,numberOfNeighborhoods)
    c=[]
    indices=set(indices)
    for t in tiles:
        t=set(t)
        c.append(list(t.intersection(indices)))
    c=np.array([np.array(x) for x in c if x != []])
    return c

def splitImageIntoNRegions(height,width,channels,numberOfTiles):
    indArray=[]
    ind=0
    for i in range(height):
        colArray=[]
        for j in range(width*channels):
            colArray.append(ind)
            ind=ind+1
        indArray.append(colArray)
    vTiles=np.array_split(np.asarray(indArray),numberOfTiles)
    hTiles=np.array_split(np.asarray(indArray),numberOfTiles,axis=1)
    hTiles=[[i for t in tiles for i in t] for tiles in hTiles] 
    vTiles=[[i for t in tiles for i in t] for tiles in vTiles]
    chunks=[[] for _ in range(len(hTiles)*len(vTiles))]
    ci=0
    for vi,vt in enumerate(vTiles):
        for hi,ht in enumerate(hTiles):
            for hii in set(vt).intersection(set(ht)):
                if hii%channels==0:
                    chunks[ci].extend([hii+c for c in range(channels)])
            ci=ci+1
    return chunks
    
def splitImageIntoNbyNRegions(height,width,channels,k):
    indArray=[]
    ind=0
    for i in range(height):
        colArray=[]
        for j in range(width):
            channelArray=[]
            for c in range(channels):
                channelArray.append(ind)
                ind=ind+1
            colArray.append(channelArray)
        indArray.append(colArray)
    chunks=[]
    y=0
    for rows in range(0,height,k):
        if rows+k>height:
            r=indArray[rows:]
        else:
            r=indArray[rows:rows+k]
        if r:
            for cols in range(0,width,k):
                chunks.append([])
                for rr in r:
                    if cols+k>width:
                        ch=[s for c in rr[cols:] for s in c]
                    else:
                        ch=[s for c in rr[cols:cols+k] for s in c]
                    if ch:
                        chunks[-1].extend(ch)
            y=y+1
    return chunks
    
def getRBGMatrix(image):
    rgb=[]
    for i,x in enumerate(image):
        if i%3==0:
            rgb.append([image[i],image[i+1],image[i+2]])
        else:
            continue
    rgb=np.reshape(rgb,(32,32))
    return rgb
    
def unequalChunks(l,n,i):
    if len(l)-n < i+n :
        return l[i:]
    else:
        return l[i:i+n]
        
def sigmoid(x):  
    return exp(-np.logaddexp(0, -x))

def softmax(x):
    #TO PREVENT OVERFLOWING OF EXP
    x=[709 if i > 709 else i for i in x]
    return np.exp(x) / np.sum(np.exp(x), axis=0)
    
def is_ascii(s):
    return all(ord(c) < 128 for c in s)    

def toBin(ind,sequenceLength):
    lst=[0]*(sequenceLength-1)
    lst[ind]=1
    return lst  

def fromBin(lst):    
    encoding=lst.index(1)
    return encoding    

def euclidean(lst1,lst2):
    dis=np.sqrt(np.square(np.subtract(lst1,lst2)))
    return dis
    
def list_duplicates_of(seq,item):
    return [(i,x) for i, x in enumerate(seq) if np.round(x,decimals=4)==np.round(item,decimals=4)]
            
def findDivisors(n):
    divs = []
    for i in range(2,int(np.sqrt(n))+1):
        if n%i == 0:
            divs.extend([i,n/i])
    divs= sorted(list(set(divs)))
    for d in divs:
        yield d
        
def reduceSearchSpace(X,searchSpaces):
    reduced=np.zeros(len(searchSpaces))
    for iss in range(len(searchSpaces)):
        x=np.zeros(len(searchSpaces[iss]))
        for i in range(len(searchSpaces[iss])):
            x[i]=X[searchSpaces[iss][i]]
        reduced[iss]=np.mean(x)
    return reduced
        
def findPowersOf2(start,end):
    pows = []
    i=1
    while 2**i <= end:
        if 2**i >= start:
            pows.append(2**i)
        i=i+1
    return pows
    
def doublesOfN(n,limit):
    lst=[]
    i=0
    while n*(2**i) <= limit:
        lst.append(n*(2**i))
        i=i+1
    return lst
    
def halfsOfN(n,limit):
    lst=[n]
    while n/2 >= limit:
        n=int(n/2)
        lst.append(n)
    return lst
    
def findDifferentPixels(image1,image2):
    inds=[i for i,_ in enumerate(image1) if not image1[i] == image2[i]]
    return inds
    
def findSimilarPixels(image1,image2):
    inds=[i for i,_ in enumerate(image1) if image1[i] == image2[i]]
    return inds