import numpy as np
from generateGLV import *

class TS_HGLV(TS_GLV):
    def __init__(self, numberSpecies, numberOfExperiments, noisePar, genPar, pertuPar, timestep_=0.01):
        self.numberSpecies = numberSpecies
        self.numberOfExperiments = numberOfExperiments
        
        self.genPar = genPar
        self.noisePar = noisePar
        self.pertuPar = pertuPar
        
        # Steady State. For now this is just scaled. As is discussed before. 
        self.isScaled = True
        self.steadystate = np.ones(self.numberSpecies)
        
        # Set parameters just equal to zero for now.
        self.growth = np.zeros((self.numberOfExperiments,self.numberSpecies))
        self.interMatrix = np.zeros((self.numberOfExperiments,self.numberSpecies,self.numberSpecies))
        self.secInterMatrix = np.zeros((self.numberOfExperiments,self.numberSpecies,self.numberSpecies))
        self.parametersAreGen = False
        
        
        # Set seed for random
        self.seedGen = seedList2[:self.numberOfExperiments]
        
        # Generate random matrix
        self.generateParameter()
        # Generate random initial
        self.initialStates = self.pertubation(True) # Seed for initial will follow up on the seed of generateParameter
        
        
        Timeseries.__init__(self,self.HGLV, self.initialStates, 
                            noiseType=self.noisePar["noiseType"], 
                            noiseStrength=self.noisePar["noiseStrength"],
                            timestep = timestep_)
        
    
    
    
    def generateParameter(self):
        self.numberOfAttempts = np.zeros(self.numberOfExperiments)
        for experiment in range(self.numberOfExperiments):
            np.random.seed(self.seedGen[experiment])
            
            IsStable = False
            attempts = 0 
            
            while not IsStable:
                if self.isScaled:# Parameters size is steady state depended! 
                    
                    newMatrix = self.genPar["interStrenghtFirst"]*np.random.randn(self.numberSpecies,self.numberSpecies)
                    selfInter = np.random.uniform(-1.9,-0.1,size=self.numberSpecies) # Keystone
                    np.fill_diagonal(newMatrix, selfInter)
                    
                    newSecMatrix = self.genPar["interStrenghtSecond"]*np.random.randn(self.numberSpecies,self.numberSpecies)
                    np.fill_diagonal(newSecMatrix, 0) # Second order self interaction = 0.
                    
                    newGrowth = - self.steadystate@newMatrix - self.steadystate**2@newSecMatrix
                
                    IsStable = self.isStable(newMatrix, newSecMatrix, self.steadystate)
                else:
                    print("Error: Non scaled is not supported yet!")
                    return
                attempts += 1

            self.growth[experiment] = newGrowth
            self.interMatrix[experiment] = newMatrix
            self.secInterMatrix[experiment] = newSecMatrix
        self.parametersAreGen = True
        
        # After everything is generated Let's construct the beta matrix
        self.constructBeta()
        
    
    def constructBeta(self):
        tempBeta = np.zeros(shape=(self.numberOfExperiments,2*self.numberSpecies+1,self.numberSpecies))
        # growthrate:
        tempBeta[:,0,:] = self.growth
        tempBeta[:, 1:self.numberSpecies+1, :] = self.interMatrix
        tempBeta[:, self.numberSpecies+1:, :] = self.secInterMatrix
        self.beta = tempBeta
    
    def isStable(self,interMatrix, secInterMatrix, steady):
        # J = M.T@diag(s) + 2*s.T@s*N
        J = interMatrix.T@np.diag(steady) + 2 * (steady.reshape(-1,1)@steady.reshape(1,-1))*secInterMatrix 
        if np.any(np.real(np.linalg.eigvals(J)) > 0):
            return False
        else:
            return True
    
    def HGLV(self, currentState):
        if self.parametersAreGen == False:
            print("Error: model parameters have not been generated")
            return
        x = np.append(1,np.append(currentState,currentState**2))
        return currentState*(x@self.beta[self.e])