# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 21:34:12 2019
Particles
@author: rayanmosli
"""
import numpy as np
from copy import deepcopy
from math import cos,pi
import gc
from keras.backend import clear_session
np.random.seed(0)
#from Utilities import reduceSearchSpace

class particle:
    def __init__(self,particleid=0):
        self.particleID=particleid
        self.bestFitness=0
        self.pastFitness=[]
        self.currentPosition=None
        self.nextPosition=None
        self.bestPosition=None
        self.currentVelocity=None
        self.currentFitness=0
        self.nextVelocity=None
        self.blocks={}
    
    def setNextPosition(self,newPosition):
        self.nextPosition=deepcopy(newPosition)
            
    def setCurrentPosition(self,newPosition):
        self.currentPosition=deepcopy(newPosition)
            
    def setBestPosition(self,newPosition):
        self.bestPosition=deepcopy(newPosition)
            
    def setBestFitnessScore(self,newScore):
        self.bestFitness=newScore
        
    def setCurrentFitnessScore(self,newScore):
        self.currentFitness=newScore
                
    def setW(self):
        self.wEND=0.0
        self.wSTART=1.0
        
    def cleanParticle(self):
        del self.particleID
        del self.currentPosition
        del self.nextPosition
        del self.bestPosition
        del self.bestFitness
        del self.currentVelocity
        del self.currentFitness
        clear_session()
        gc.collect()

    def push(self,fitness,indices):
        self.pastFitness.append((fitness,indices))   
        self.pastFitness.sort(key=lambda x: x[0],reverse=True)
        
    def popHigh(self):
        indices=deepcopy(self.pastFitness[0][1])
        fitness=self.pastFitness[0][0]
        self.pastFitness=self.pastFitness[1:]
        return fitness,indices
        
    def peekHigh(self):
        if self.pastFitness:
            return self.pastFitness[0][0]
        else:
            return 0
    
    def popLow(self):
        indices=deepcopy(self.pastFitness[-1][1])
        fitness=self.pastFitness[-1][0]
        self.pastFitness=self.pastFitness[:-1]
        return fitness,indices
        
    def peekLow(self):
        if self.pastFitness:
            return self.pastFitness[-1][0]
        else:
            return 0
    
    def printParticleInformation(self):
        print('Particle %s -- Best Fitness %s \n'%(str(self.particleID),str(self.bestFitness)))      

    def Velocity(self,swarmBestPosition,q,C1,C2,maxQueries,searchSpaces,maxChange):
         self.W=self.calculateW(q,swarmBestPosition,self.wSTART,self.wEND,maxQueries)
         particleBestDelta=np.multiply(np.multiply(np.subtract(self.bestPosition,self.currentPosition),np.random.uniform(0.0,1.0,len(self.bestPosition))),C2)
         swarmBestDelta=np.multiply(np.multiply(np.subtract(swarmBestPosition,self.currentPosition),np.random.uniform(0.0,1.0,len(swarmBestPosition))),C1)
         deltas=np.add(np.asarray(particleBestDelta),np.asarray(swarmBestDelta))
         v=np.add(self.W*np.clip(self.currentVelocity,-1*maxChange,maxChange) , deltas) 
         self.nextVelocity=deepcopy(v)
         self.currentVelocity=deepcopy(v)
         return v
         
    def calculateNextPosition(self,swarmBestPosition,T,C1,C2,inputX,searchSpaces,maxChangeLower,maxChangeUpper,maxQueries,maxChange,lowerBound=-0.5,upperBound=0.5):
        v=self.Velocity(swarmBestPosition,T,C1,C2,maxQueries,searchSpaces,maxChange)
        
        self.Perturb(v,maxChangeLower,maxChangeUpper,lowerBound,upperBound,inputX)
        self.setCurrentPosition(self.nextPosition)      
        return v

    def calculateK(self,q,maxQueries):
        constrictionFactor=((cos((pi/maxQueries)*q))+2.5)/4
        return constrictionFactor
        
    def calculateW(self,q,swarmBestPosition,wSTART,wEND,maxQueries):
        if np.all(np.equal(self.bestPosition,swarmBestPosition)):
            W=wEND
            return W
        elif not np.all(np.equal(self.bestPosition , swarmBestPosition)):
            W=wEND+((wSTART-wEND)*(1-(q/maxQueries)))
            return W
            
    def Perturb(self,indices,maxChangeLower,maxChangeUpper,lowerBound,upperBound,inputX):
        self.nextPosition=np.clip(np.clip(np.add(indices,self.currentPosition),maxChangeLower,maxChangeUpper),lowerBound,upperBound)
            
    def getPerturbParameters(self, patches,inputX):    
        indices=[0]*len(inputX)
        for iss,vel in patches:
            for i in iss:
                indices[i]=vel
        return indices