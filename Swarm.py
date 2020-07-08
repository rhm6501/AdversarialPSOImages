# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 21:34:27 2019
Swarm
@author: rayanmosli
"""
import random
import numpy as np
from copy import deepcopy
from Utilities import predictSample,reshapeImage,compareImages,splitImageIntoNbyNRegions,halfsOfN
from particle import particle
import gc
from keras.backend import clear_session
from Dataset import returnDimensions
from itertools import product
np.random.seed(0)
random.seed(0)

    
class Swarm:
    bestParticleID=-1
    C1=0.0
    C2=0.0
    initialFitness=0.0
    dLength=None
    numOfQueries=0
    bestFitness=0
    granularityCurrentLevel=0
    
    def __init__(self,numOfParticles,model,ind,maxChange,dataset,dLength,verbosity,topN,targeted,queries):
        self.numberOfParticles=numOfParticles
        self.targetModel=model
        self.classIND=ind
        self.dLength=dLength
        self.maxChange=maxChange
        self.dataset=dataset
        self.verbosity=verbosity
        self.topN=topN
        self.targeted=targeted
        self.queries=queries
        
    def cleanSwarm(self):
        del self.numberOfParticles
        del self.bestFitness
        del self.bestPosition
        del self.currentPosition
        del self.inputX
        del self.Particles
        clear_session()
        gc.collect()
        
    def showImage(self,x):
        from Graphics import Graphics
        g=Graphics(self.dataset)
        x=reshapeImage(x,self.dataset)
        g.show(x,self.swarmLabel,save=False)
        
    def setCurrentPosition(self,newPosition):
        self.currentPosition=deepcopy(newPosition)
            
    def setBestPosition(self,newPosition):
        self.bestPosition=deepcopy(newPosition)

    def setInitialFitness(self,initialFitness):
        self.initialFitness=initialFitness
        
    def setCs(self,C1,C2):       
        self.C1=C1
        self.C2=C2
    
    def getImageDimensions(self):
        self.height,self.width,self.channels=returnDimensions(self.dataset)
        
    def setSwarmAttributes(self,x,C1,C2,lowerBoundary,upperBoundary,blockSize):
        self.setCs(C1,C2)
        self.setBestPosition(x)
        self.setCurrentPosition(x)
        self.setInputX(x)
        self.setBoundaries(lowerBoundary,upperBoundary)
        self.maxChangeLower=np.subtract(self.inputX,self.maxChange)
        self.maxChangeUpper=np.add(self.inputX,self.maxChange)
        self.blockSize=blockSize
        
    def setBoundaries(self,lowerBoundary,upperBoundary):
        self.lowerBoundary=lowerBoundary
        self.upperBoundary=upperBoundary
        
    def setBestFitnessScore(self,newScore):
        self.bestFitness=newScore
        
    def reshapeAndCompare(self,positions):
        z=reshapeImage(positions,self.dataset)
        diff=compareImages(self.inputX,positions,len(positions))
        return z,diff
        
    def returnTopNPred(self,pred):
        if self.topN > 1:
            pred=np.argsort(pred)[:][::-1][:self.topN]
        elif self.topN==1:
            pred=np.argmax(pred)
        return pred
        
    def setInputX(self,newPosition):
        self.inputX=deepcopy(newPosition)
            
    def calculateBaselineConfidence(self,model,sample,ind):
        proba=predictSample(model,sample)[0]
        self.bestProba=proba
        self.numOfQueries=self.numOfQueries+1
        if type(ind) is np.int64 or type(ind) is int:
            proba=proba[ind]
        elif type(ind) is np.ndarray:
            avgProba=[]
            for i,x in enumerate(ind):
                avgProba.append(proba[x])
            proba=avgProba
        self.baselineConfidence=proba
        return proba
        
    def fitnessScore(self,test):
        proba=predictSample(self.targetModel,test)[0]
        self.numOfQueries=self.numOfQueries+1
        tempProba=proba
        if type(self.classIND)==np.int64 or type(self.classIND) is int:
            proba=proba[self.classIND]
            if self.targeted==False:
                proba=self.baselineConfidence-proba
            elif self.targeted==True:
                proba=proba-self.baselineConfidence
        elif type(self.classIND)==np.ndarray:
            avgProba=[]
            if self.targeted==False:
                for i,x in enumerate(self.classIND):
                    avgProba.append(proba[x])
                avgProba=sum([base-prob for base,prob in zip(self.baselineConfidence,avgProba)])
            elif self.targeted==True:
                avgProba.append(proba[self.classIND[0]]-self.baselineConfidence[0])
                avgProba.extend([base-proba[x] for base,prob in zip(self.baselineConfidence[1:],self.classIND[1:])])
                avgProba=sum(avgProba)
            proba=avgProba
        fitness=proba
        return fitness,tempProba
       
    def generateISSIndicesDictionary(self):
        self.issIndices={}
        cd=list(product([0,-1,1],repeat=self.channels))
        if self.channels==3:
            cd=[(a,b,c) for a,b,c in cd if len(np.nonzero([a,b,c])[0])==1] 
        elif self.channels==1:
            cd=[(a) for a in cd if not a[0]==0] 
        for i in range(len(self.ISS)):
            self.issIndices[i]=deepcopy(cd)
        
    def generateIndividualSearchSpaces(self):
        self.granularityLevels=halfsOfN(self.blockSize,2)
        self.ISS=splitImageIntoNbyNRegions(self.height,self.width,self.channels,self.granularityLevels[self.granularityCurrentLevel])
        self.generateISSIndicesDictionary()
        self.updateChangeRate()
        if self.verbosity>=2:
            print("Blocksize= %s"%(self.granularityLevels[self.granularityCurrentLevel]))
            print("Change Rate Per Particle= %s"%(self.changeRate))

    def increaseISSGranularity(self):
        if self.granularityCurrentLevel+1<len(self.granularityLevels):
            self.granularityCurrentLevel=self.granularityCurrentLevel+1
            self.ISS=splitImageIntoNbyNRegions(self.height,self.width,self.channels,self.granularityLevels[self.granularityCurrentLevel])
            self.generateISSIndicesDictionary()
            self.Particles=self.initializeParticles(self.bestPosition)
            self.Check()
            if self.verbosity>=2:
                print("Blocksize= %s"%(self.granularityLevels[self.granularityCurrentLevel]))
                print("Change Rate Per Particle= %s"%(self.changeRate)) 
        else:
            if self.granularityCurrentLevel+1==len(self.granularityLevels):
                self.generateISSIndicesDictionary()
                self.resetParticles()
                self.Check
        return
       
    def updateChangeRate(self):
        lenOfIndices=len(self.issIndices)
        self.changeRate=int(np.ceil(lenOfIndices/self.numberOfParticles))
        
    def initializeSwarm(self,sample,fitness):
        self.Particles=[particle]*self.numberOfParticles
        self.setBestPosition(sample)
        self.setCurrentPosition(sample)
        self.setBestFitnessScore(self.initialFitness)

    def initializeSwarmAndParticles(self,sample,fitness):
        self.pastFitness=[]
        self.getImageDimensions()
        self.flag=False
        if self.targeted==True:
            self.swarmLabel=-1
        else:
            self.swarmLabel=self.classIND
        self.previousGranBest=deepcopy(self.inputX)
        self.initializeSwarm(sample,fitness)
        self.generateIndividualSearchSpaces()
        self.Particles=self.initializeParticles(sample)
        self.Check()
        self.pastFitness.append(self.bestFitness)
        if self.verbosity>=2:
            print('After Initialization - Diff with input %s - Best Fitness %s - Number of Queries %s'%(str(np.sqrt(compareImages(self.inputX,self.bestPosition,len(self.bestPosition))*self.dLength)),self.bestFitness,str(self.numOfQueries)))
        
        
    def initializeParticles(self,startingPosition):
        particleList=[]
        for particles in range(0,self.numberOfParticles):
            p=None
            p=particle(particles)
            p.setW()
            p.blocks={}
            p.setCurrentPosition(startingPosition)
            p.setBestPosition(startingPosition)
            p.setNextPosition(startingPosition)
            p.currentVelocity=[0]*self.dLength
            p,newProba,_=self.randomizeParticle(startingPosition,p)
            particleList.append(deepcopy(p))
            del p
        return particleList
        
    def moveParticleAndCalNewFitness(self,p):
        v=p.calculateNextPosition(self.bestPosition,self.numOfQueries,self.C1,self.C2,self.inputX,
                                               self.ISS,self.maxChangeLower,self.maxChangeUpper,self.queries,self.maxChange,
                                               lowerBound=self.lowerBoundary,upperBound=self.upperBoundary)
        z,_=self.reshapeAndCompare(p.nextPosition)
        newFitness,newProba=self.fitnessScore(z)
        p.push(newFitness-p.currentFitness,v)
        p.setCurrentFitnessScore(newFitness)
        self.checkNewParticlePosition(p,newFitness,newProba)
        return newFitness,newProba

    def randomizeParticle(self,basePosition,p):
        iss=[]
        iss=self.returnRandomSearchSpaces(p)
        if not iss:
            return p,0,p.currentFitness
        indices=self.getRandomizationParameters(iss)
        temp=deepcopy(basePosition)
        p.currentPosition=self.Randomize(indices,temp)
        z,_=self.reshapeAndCompare(np.array(p.currentPosition))
        newFitness,newProba=self.fitnessScore(z)
        p.push(newFitness-p.currentFitness,indices)
        p.setCurrentFitnessScore(newFitness)
        self.checkNewParticlePosition(p,newFitness,newProba)
        del temp
        return p,newProba,newFitness    
    
    def checkNewParticlePosition(self,p,newFitness,newProba,oldPosition=[],velocity=[]):
        if newFitness>p.bestFitness:
            p.setBestFitnessScore(newFitness)
            p.setBestPosition(p.currentPosition)
        if p.bestFitness > self.bestFitness:
            self.setBestFitnessScore(p.bestFitness)
            self.setBestPosition(p.bestPosition)
            self.bestParticleID=p.particleID
            self.bestProba=newProba
        
    def Randomize(self,indices,temp):
        temp=np.clip(np.clip(np.add(indices,temp),self.maxChangeLower,self.maxChangeUpper),self.lowerBoundary,self.upperBoundary)
        return temp
    
    def returnRandomSearchSpaces(self,p):
        if len(p.blocks)==len(self.ISS):
            p.blocks={}
        keys=random.sample(list(self.issIndices.keys()),self.changeRate if self.changeRate<=len(self.issIndices) else len(self.issIndices))
        iss=self.iterateKeys(p,keys)
        if len(self.issIndices)==0:
            self.flag=True
        if len(iss)==0:
            return []
        else:
            return iss

    def iterateKeys(self,p,keys):
        iss={}
        for key in keys:
            if len(iss)>=self.changeRate:
                break
            if len(self.issIndices[key])==0:
                self.issIndices.pop(key,None)
                continue
            if key in p.blocks:
                keysInParticle=deepcopy(p.blocks[key])
                if self.channels==3:
                    keysInParticle=[tuple(np.multiply((a,b,c),d)) for a,b,c in keysInParticle for d in [1,-1]] 
                elif self.channels==1:
                    keysInParticle=[tuple(np.multiply(a,d)) for a in keysInParticle for d in [1,-1]] 
                directionsToSample=list(set(self.issIndices[key]).difference(keysInParticle)) 
            else:
                p.blocks[key]=[]
                directionsToSample=self.issIndices[key]
            if not directionsToSample:
                continue
            lowLevel=random.choice(directionsToSample)
            iss[key]=lowLevel
            p.blocks[key].append(lowLevel)
            self.issIndices[key].remove(lowLevel)
            if len(self.issIndices[key])==0:
                self.issIndices.pop(key,None)
        return iss
    
    def getRandomizationParameters(self,searchSpace):
         indices=[0]*self.dLength
         for iss in searchSpace:
             directions=searchSpace[iss]
             step=1.0
             for i in self.ISS[iss]:
                indices[i]=directions[i%self.channels]*self.maxChange*step
         return indices

    def backTrack(self):
        if self.verbosity>=2:
            print('Back-tracking...')
        startingFitness=self.bestFitness
        startingQueries=self.numOfQueries
        temp=deepcopy(self.bestPosition)
        for p in self.Particles:
            while p.peekLow()<0:
                oldTemp=deepcopy(temp)
                fitness,indices=p.popLow()
                temp=np.clip(np.clip(np.subtract(temp,indices),self.maxChangeLower,self.maxChangeUpper),self.lowerBoundary,self.upperBoundary)
                z,_=self.reshapeAndCompare(temp)
                newFitness,newProba=self.fitnessScore(z)
                if newFitness>=self.bestFitness:
                    self.bestFitness=newFitness
                    self.bestPosition=deepcopy(temp)
                    self.Check()
                    if self.labelCheck() or self.numOfQueries>=self.queries:
                        if self.verbosity>=2:
                            print("Back-tracking improved fitness by %s using %s Queries"%(self.bestFitness-startingFitness, self.numOfQueries-startingQueries))
                        return
                else:
                    temp=deepcopy(oldTemp)
        if self.verbosity>=2:
            print("Back-tracking improved fitness by %s using %s Queries"%(self.bestFitness-startingFitness, self.numOfQueries-startingQueries))
        
    def Move(self):
        for p in self.Particles:
            newFitness,newProba=self.moveParticleAndCalNewFitness(p)

    def resetParticles(self):
            for p in self.Particles:
                _,_,fitness=self.randomizeParticle(p.currentPosition,p)

    def runSearch(self):
        self.Move()
        self.Check()
        if self.labelCheck() or self.numOfQueries>=self.queries:
            return
        if self.flag==True:
            self.backTrack()
            if self.labelCheck() or self.numOfQueries>=self.queries:
                return
            self.increaseISSGranularity()
            self.flag=False
        else:
            self.resetParticles()
        self.pastFitness.append(self.bestFitness)

    def Check(self):
        self.swarmLabel=self.returnTopNPred(self.bestProba)
        
    def searchOptimum(self):
        if self.labelCheck():
            return self.bestPosition , self.bestFitness,0,self.numOfQueries
        iteration=0
        while self.queries>self.numOfQueries:
            if self.labelCheck() or self.numOfQueries>=self.queries:
                return self.bestPosition , self.bestFitness, iteration ,self.numOfQueries
            self.runSearch()
            if self.verbosity>=2:
                print('Iteration %s - Diff with input %s - Best Fitness %s - Number of Queries %s'%(str(iteration+1),str(np.sqrt(compareImages(self.inputX,self.bestPosition,len(self.bestPosition))*self.dLength)),self.bestFitness,str(self.numOfQueries)))
            iteration=iteration+1
        return self.bestPosition , self.bestFitness, iteration ,self.numOfQueries               
                
    def labelCheck(self):
        if self.targeted==True:
            if not 'imagenet' in self.dataset and self.swarmLabel==self.classIND:
                return True
            elif 'imagenet' in self.dataset:
                if self.topN==1:
                    if self.swarmLabel==self.classIND:
                        return True
                    else:
                        return False
                elif self.topN>1:
                    if not any(elem in self.classIND[1:] for elem in self.swarmLabel[1:]) and self.swarmLabel[0]==self.classIND[0]:
                        return True
                    else:
                        return False
            else:
                return False
        elif self.targeted==False:
            if self.topN==1:
                if self.swarmLabel==self.classIND:
                    return False
                elif not self.swarmLabel==self.classIND:
                    return True
            elif self.topN>1:
                if any(elem in self.classIND for elem in self.swarmLabel):
                    return False
                else:
                    return True
