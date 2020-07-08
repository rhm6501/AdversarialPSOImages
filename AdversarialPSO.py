import argparse
    
parser = argparse.ArgumentParser(description='PSO Parameters')
parser.add_argument('--dataset', '-d' , type=str, help='Supports CIFAR10, MNIST, imagenet224, and imagenet299',default='CIFAR10')
parser.add_argument('--maxChange', type=float, help='Controls the L-infinity distance between the source and destination images',default=8/255)
parser.add_argument('--numOfParticles', '-p' , type=int, help='Number of particles in the swarm',default=5)
parser.add_argument('--targeted', '-t' , help='Choose random target when crafting examples',default=False,action='store_true')
parser.add_argument('--C1' , type=float, help='Controls exploitation weight',default=2.0)
parser.add_argument('--C2' , type=float, help='Controls explorations weight',default=2.0)
parser.add_argument('--Samples' , '-n',type=int, help='Number of test Samples to attack',default=1000)
parser.add_argument('--Randomize',help='Randomize dataset',default=False,action='store_true')
parser.add_argument('--verbose','-v',type=int,help='Verbosity level. 0 for no terminal logging, 1 for samples results only, and 2 for swarm level verbosity',default=2)
parser.add_argument('--topN',type=int,help='Specify the number of labels to reduce when attacking imagenet',default=1)
parser.add_argument('--sample',type=int,help='Specify which sample to attack',default=-1)
parser.add_argument('--blockSize',type=int,help='Initial blocksize for seperating image into tiles',default=8)
parser.add_argument('--Queries','-q',type=int,help='Mazimum number of queries',default=10000)
parser.add_argument('--pars',help='Run in Parsimonious... samples',default=False,action='store_true')

args = parser.parse_args()

maxChange=args.maxChange
numOfParticles=args.numOfParticles
targeted=args.targeted
N=args.Samples
verbosity=args.verbose
C1=args.C1
C2=args.C2
Randomize=args.Randomize
dataset=args.dataset
sample=args.sample
blockSize=args.blockSize
queries=args.Queries
pars=args.pars

import random
import numpy as np
import os
from keras.backend import clear_session
from Dataset import loadImageNet224Model,loadImageNet299Model,loadCIFARModel,loadMNISTModel,Dataset
from Utilities import predictSample,flattenImage,calculateDiffAndPredict,compareImagesL0,compareImagesLinfty
from Graphics import Graphics
from Swarm import Swarm
import gc

if not 'imagenet' in dataset and args.topN>1:
    topN=1
    print('Top N only supports attacks on imagenet. Resetting to 1\n')
else:
    topN=args.topN
    
correctlyClassified=0

if dataset == 'CIFAR10':
    dLength=32*32*3
    plotDirs=os.path.join('.','Results','Plots_CIFAR')
elif dataset == 'MNIST':
    dLength=28*28
    plotDirs=os.path.join('.','Results','Plots_MNIST')
elif dataset == 'imagenet224':
    dLength=224*224*3
    plotDirs=os.path.join('.','Results','Plots_ImageNet')
elif dataset == 'imagenet299':
    dLength=299*299*3
    plotDirs=os.path.join('.','Results','Plots_ImageNet')
       
    
def prepareLogFilesAndOutputDirectoriy():
    if not os.path.isdir(os.path.join('.','Results')):
        os.mkdir(os.path.join('.','Results'))
    if not os.path.isdir(plotDirs):
        os.mkdir(plotDirs)
    with open(os.path.join('.','Results',dataset+'_PSO_Results.csv'),'w') as f:
        f.write('Sample,BaselineCofidence,BaselineFitness,TargetLabel,Prediction_Before_PSO, Confidence_After_PSO,Fitness_After_PSO,Prediction_After_PSO,Iteration,L2_Difference_Between_Images,L0,LInfinity,Number_of_Model_Queries,Results\n')

def checkPredicition(pred,y):
    if not ((np.argmax(pred)==np.argmax(y)and(dataset=='MNIST' or dataset=='CIFAR10'))or (np.argmax(y) in np.argsort(pred)[:][::-1][:3]) and 'imagenet' in dataset):
        return False
    global correctlyClassified
    correctlyClassified=correctlyClassified+1
    return True


def getModelandX(x):
    if dataset=='imagenet224':
        x=np.divide(x,255.0)
        x=np.subtract(x,0.5)
        x=np.multiply(x,2)
        x=np.asarray(x)
        model=loadImageNet224Model()
    if dataset=='imagenet299':
        x=np.subtract(x,0.5)
        x=np.multiply(x,2)
        x=np.asarray(x)
        model=loadImageNet299Model()
    elif dataset=='CIFAR10':
        model=loadCIFARModel()
    elif dataset=='MNIST':
        model=loadMNISTModel()
    x,lowerBoundary,upperBoundary=getBoundaries(x)
    return model,x,lowerBoundary,upperBoundary

def getBoundaries(x):
    if dataset=='CIFAR10' or dataset=='MNIST':
        if np.min(x)<0:
            lowerBoundary=-0.5
            upperBoundary=0.5
        else:
            lowerBoundary=0.0
            upperBoundary=1.0
            x=np.add(x,0.5)
    elif 'imagenet' in dataset:
        lowerBoundary=-1.0
        upperBoundary=1.0
    return x,lowerBoundary,upperBoundary
    
def createPlotPath(i):
    dirPath=os.path.join(plotDirs,str(i))
    if not os.path.isdir(dirPath):
        os.mkdir(dirPath)
    return dirPath

def pseudorandom_target(index, total_indices, true_class):
    rng = np.random.RandomState(index)
    target = true_class
    while target == true_class:
        target = rng.randint(0, total_indices)
    return target

def getTargetLabel(pred,i):
    if targeted==True and topN == 1:
        targetLabel=pseudorandom_target(i,len(pred[0]),np.argmax(pred))
    elif targeted==True and topN > 1:
        targetLabel=list(set(g.getClassName().keys()).difference(np.argsort(pred[0])[:][::-1][:topN]))
        targetLabel=[random.choice(targetLabel)]
        targetLabel.extend(np.argsort(pred[0])[:][::-1][:topN-1])
    else:
        if topN>1:
            targetLabel=np.argsort(pred[0])[:][::-1][:topN]
        elif topN==1:
            targetLabel=np.argmax(pred)
    return targetLabel
        
  
def Initialization(pred,x,model,i,lowerBoundary,upperBoundary):
    dirPath=createPlotPath(i)
    targetLabel=getTargetLabel(pred,i)
    swarm=Swarm(numOfParticles,model,targetLabel,maxChange,dataset,dLength,verbosity,topN,targeted,queries)  
    g.show(x,swarm.returnTopNPred(pred),save=True,path=os.path.join(dirPath,'Before.png'))
    pred=swarm.returnTopNPred(pred[0])
    baselineConfidence=swarm.calculateBaselineConfidence(model,x,targetLabel)
    x=flattenImage(x)
    swarm.setSwarmAttributes(x,C1,C2,lowerBoundary,upperBoundary,blockSize)
    initialFitness=0
    swarm.setInitialFitness(initialFitness)
    if verbosity>=1:            
        print('Model Prediction Before PSO= %s'%(pred))
        print('Baseline Confidence= %s'%(str(baselineConfidence)))
        print('Baseline Fitness= %s\n'%(str(initialFitness)))
    swarm.initializeSwarmAndParticles(x,initialFitness) 
    return swarm,baselineConfidence,pred,initialFitness,targetLabel,dirPath,x
    
def adversarialPSO(x,y,g,i):
    model,x,lowerBoundary,upperBoundary=getModelandX(x)
    pred=predictSample(model,x)
    status=checkPredicition(pred,y)
    if status==False:
        print("Sample %s incorrectly classified...Skipping...\n" %(i))
        return -1,-1,-1,-1,-1
    print("Searching Advresarial Example for test sample %s...\n" %(i))
    numberOfQueries=0
    swarm,baselineConfidence,pred,initialFitness,targetLabel,dirPath,x=Initialization(pred,x,model,i,lowerBoundary,upperBoundary)
    _,_,iterations,numberOfQueries=swarm.searchOptimum()
    if verbosity>=1:
        print('\nBest Fitness Score= %s'%(swarm.bestFitness))
    diff,z,predAfter=calculateDiffAndPredict(x,swarm.bestPosition,model,dataset)
    finalFitness=swarm.bestFitness
    g.show(z,swarm.swarmLabel,save=True,path=os.path.join(dirPath,'AfterPSO.png'))
    if verbosity>=1:
        print('Model Prediction After PSO= %s'%(swarm.swarmLabel))
        print('Best Fitness Score= %s'%(swarm.bestFitness)) 
        print('Difference between origial and adversarial image= %s'%(np.sqrt(diff*dLength))) 
        print('Number of Queries to model= %s\n'%(numberOfQueries))
    with open(os.path.join('.','Results',dataset+'_PSO_Results.csv'),'a') as f:
        f.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n'%(str(i),str(baselineConfidence).replace(',', ' '),str(initialFitness),str(targetLabel).replace(',', ' '),str(pred).replace(',', ' '),str(np.sum(baselineConfidence)-finalFitness),str(finalFitness),str(swarm.swarmLabel).replace(',', ' '),str(iterations),str(np.sqrt(diff*dLength)),str(compareImagesL0(swarm.bestPosition,swarm.inputX)),str(compareImagesLinfty(swarm.bestPosition,swarm.inputX)),str(numberOfQueries),str(1 if swarm.labelCheck() else 0)))
    bf=swarm.bestFitness
    bp=swarm.bestPosition
    swarm.cleanSwarm()
    del swarm  
    clear_session()
    del model
    gc.collect()
    return bf,bp,diff,numberOfQueries,iterations
            
if __name__ == "__main__":
    g=Graphics(dataset)
    prepareLogFilesAndOutputDirectoriy()
    testData,testLabels=Dataset(dataset,pars=pars,targeted=targeted)
    if Randomize==True:
        c=list(zip(testData,testLabels))
        random.shuffle(c)
        testData,testLabels = zip(*c)
    i=0
    if sample == -1:
        for x,y in zip(testData,testLabels):
            if correctlyClassified>=N:
                break
            bf,bp,diff,numberOfQueries,iterations=adversarialPSO(x,y,g,i)
            clear_session()
            gc.collect()
            i=i+1
    elif sample >= 0:
            bf,bp,diff,numberOfQueries,iterations=adversarialPSO(testData[sample],testLabels[sample],g,sample)
            clear_session()
            gc.collect()