# This script is based on the .ipynb script TimeSeries in the preFWO folder

import numpy as np
import matplotlib.pyplot as plt
import itertools as it

seedList1 = [  785, 38169, 44682, 15570, 13274, 44387, 28742, 34599, 39125, 18973]
seedList2 = [13367,  5901,  8258,  2184, 39489, 11901, 21542, 25640, 12128, 11222]
seedList1 = np.load("seedList1.npy") # unique numbers that were generated via 
seedList2 = np.load("seedList2.npy") # np.random.randint(minInt,maxInt,size=(length))

class Timeseries():
    def __init__(self, F, initialStates, timestep=0.01, tMax=100, integrationType = "Euler" 
                 ,noiseType = "LangevinLinear", noiseStrength = 0.01):
        self.typeTimeseries = "function"
        self.isGenerated = False
        
        # System setup
        self.F = F
        self.initialStates = initialStates
        self.numberOfExperiments, self.dimension = self.initialStates.shape if len(self.initialStates.shape)==2 else (1,len(self.initialStates))
        self.timestep = timestep
        self.tMax = tMax
        self.numberOfPoints = int(float(self.tMax)/float(self.timestep))
        
        # Save setup
        #self.time = np.arange(0,self.tMax,self.timestep)
        self.result = np.zeros(shape = (self.numberOfExperiments,self.numberOfPoints,self.dimension))
        self.result[:,0,:] = self.initialStates
        self.modelDiff = np.zeros(shape = (self.numberOfExperiments,self.numberOfPoints,self.dimension))
        self.noiseDiff = np.zeros(shape = (self.numberOfExperiments,self.numberOfPoints,self.dimension))
        #self.noiseContribution = self.result - self.modelContribution
        
        # Noise parameters
        self.seeds = seedList1[:self.numberOfExperiments]
        self.typeNoise = noiseType
        self.noiseParameters = {"strenght": noiseStrength}
        
        # Call method the method that sets the correct self.nextStepDiff method. 
        # When doing numerical integration nextStepDiff will be called. 
        self.setIntegration(integrationType) # Euler or Runge-Kutta
        
        # Parameters during Generation
        self.i = 0 # Current itteration
        self.e = 0 # Current experiment
        
        
    def setIntegration(self,integrationType):
        # Set here the "nextStep" function. It is allowed to point to a function outside the class!
        # One can then parameters to the outside function with lambda operator.
        if integrationType == "Euler":
            self.integrationType = integrationType
            self.nextStepDiff = self.eulerDiff # The euler integration is just method defined here. 
        elif integrationType == "RK4":#like Runge-Kutta
            self.integrationType = integrationType
            self.nextStepDiff = self.RK4Diff
        else:
            print(f"Integration type {integrationType} is not supported")
    
    def reset(self):
        self.result = np.zeros(shape = (self.numberOfExperiments,self.numberOfPoints,self.dimension))
        self.result[:,0,:] = self.initialState
        self.modelContribution = np.zeros(shape = (self.numberOfExperiments,self.dimension))
        self.noiseContribution = np.zeros(shape = (self.numberOfExperiments,self.dimension))
    
    def generate(self):
        for experiment in range(self.numberOfExperiments):
            # set seed for the experiment
            self.e = experiment
            np.random.seed(self.seeds[experiment])
            for i in range(self.numberOfPoints-1):# -1 because we look at the next state
                self.i = i+1
                currentState = self.result[experiment,i,:]
                # Compute next step and store it immediatly
                self.modelDiff[experiment,i+1] = self.nextStepDiff(currentState)
                self.noiseDiff[experiment,i+1] = self.sampleNoise(currentState)
                self.result[experiment,i+1] = self.result[experiment,i] + self.modelDiff[experiment,i+1] + self.noiseDiff[experiment,i+1]
        self.isGenerated = True
                
    def sampleNoise(self, currentState):
        if self.typeNoise == "LangevinLinear":
            return  self.noiseParameters["strenght"] * currentState * np.sqrt(self.timestep) * np.random.normal(0, 1, self.dimension)
        
    def eulerDiff(self,currentState):
        return self.timestep*self.F(currentState)
    
    ## Runga Kutta 4th order integration
    def RK4Diff(self,currentState):
        return self.F1(currentState)/6+2/6*(self.F2(currentState)+self.F3(currentState))+self.F4(currentState)/6
    def F1(self,currentState):
        return self.timestep*self.F(currentState)
    def F2(self,currentState):
        currentState_F1=currentState+self.F1(currentState)/2
        return self.timestep*self.F(currentState_F1)
    def F3(self,currentState):
        currentState_F2=currentState+self.F2(currentState)/2
        return self.timestep*self.F(currentState_F2)
    def F4(self,currentState):
        currentState_F3=currentState+self.F3(currentState)
        return self.timestep*self.F(currentState_F3)
    
    def plot(self,experiment=None):
        time = np.arange(0,self.tMax,self.timestep)[:self.numberOfPoints]
        
        for exp  in range(self.numberOfExperiments):
            if experiment is not None: # Go to the correct experiment number
                if exp!=experiment:
                    continue
                
            for j in range(self.dimension):
                plt.plot(time,self.result[exp,:,j],label=f"Dim.{j} exp.{exp}")
            
        plt.title("Timeseries")
        plt.xlabel("Time")
        plt.ylabel("Result") 


class TS_GLV(Timeseries):
    def __init__(self, numberSpecies, numberOfExperiments, noisePar, genPar, pertuPar, timestep_=0.01, integrationType_ = "Euler"):
        self.numberSpecies = numberSpecies
        self.numberOfExperiments = numberOfExperiments
        #self.numberOfParameters = self.numberSpecies*(self.numberSpecies+1) # NxN + Nx1 number of parameters for GLV
        self.genPar = genPar
        self.noisePar = noisePar
        self.pertuPar = pertuPar
        
        try:  # Many code old code does not explicitly define "scaled", hence handle it with Try except.
            self.isScaled = self.genPar["scaled"]
        except:
            self.isScaled = True
            
        if self.isScaled is False:
            try:
                self.steadystate = self.genPar["steadystate"].astype(float)
                if len(self.steadystate)!=self.numberSpecies:
                    print("Error, given steady state has not the correct amount of species")
                    return
            except:
                print("Error, Set non-scaled but no steadystate has been provided.")
                return
        else:
            self.steadystate = np.ones(self.numberSpecies)
        
        # Set parameters just equal to zero for now.
        self.growth = np.zeros((self.numberOfExperiments,self.numberSpecies))
        self.interactionMatrix = np.zeros((self.numberOfExperiments,self.numberSpecies,self.numberSpecies))
        self.parametersAreGen = False
        
        # Set seed for random
        self.seedGen = seedList2[:self.numberOfExperiments]
        
        # Generate random matrix
        self.generateParameter()
        # Generate random initial
        self.initialStates = self.pertubation(True) # Seed for initial will follow up on the seed of generateParameter
        
        
        Timeseries.__init__(self,self.GLV, self.initialStates, 
                            noiseType=self.noisePar["noiseType"], 
                            noiseStrength=self.noisePar["noiseStrength"],
                            timestep = timestep_,
                            integrationType = integrationType_)
        
    
    def generateParameter(self):
        self.numberOfAttempts = np.zeros(self.numberOfExperiments)
        for experiment in range(self.numberOfExperiments):
            np.random.seed(self.seedGen[experiment])
            
            IsStable = False
            attempts = 0 
            while not IsStable:
                newMatrix = self.genPar["interactionStrenght"]*np.random.randn(self.numberSpecies,self.numberSpecies)
                selfInter = np.random.uniform(-1.9,-0.1,size=self.numberSpecies)
                #selfInter = -np.ones(self.numberSpecies)
                np.fill_diagonal(newMatrix, selfInter)
                
                if not self.isScaled: # Divide each column by the steady state.
                    # Very nice post on this: https://stackoverflow.com/questions/18522216
                    newMatrix = (newMatrix.T * self.steadystate**-1).T # a**-1 = divide by a.
                    
                newGrowth = - self.steadystate@newMatrix
                IsStable = self.isStable(newMatrix, self.steadystate)
                attempts += 1
            self.growth[experiment] = newGrowth
            self.interactionMatrix[experiment] = newMatrix
            self.numberOfAttempts[experiment] = attempts
        self.parametersAreGen = True
        
        # After everything is generated Lets construct the beta matrix
        self.constructBeta()
        
    def generate(self): # redefine generate to now also add pertubation;
        self.hasPerturbed = np.zeros((self.numberOfExperiments,self.numberOfPoints-1),dtype=bool)
        for experiment in range(self.numberOfExperiments):
            # set seed for the experiment
            self.e = experiment
            np.random.seed(self.seeds[experiment])
            for i in range(self.numberOfPoints-1):# -1 because we look at the next state
                self.i = i+1 # the itteration for which we now want to compute the new state
                currentState = self.result[experiment,i,:]
                # Compute next step and store it immediatly
                self.modelDiff[experiment,i+1] = self.nextStepDiff(currentState)
                self.noiseDiff[experiment,i+1] = self.sampleNoise(currentState)
                pertubation = self.pertubation(); # For the moment this is not saved.
                if not np.all(pertubation==0):
                    self.hasPerturbed[experiment, i] = True 
                
                self.result[experiment,i+1] = self.result[experiment,i] + self.modelDiff[experiment,i+1] + self.noiseDiff[experiment,i+1] + pertubation
        self.isGenerated = True
        
        # Also check which experiments are valid and which not.
        self.validExperiment = np.ones(self.numberOfExperiments,dtype="bool")
        for exp in range(self.numberOfExperiments):
            res = self.result[exp]
            if np.any(np.isnan(res)): # Overflow happend and it could not finish.
                self.validExperiment[exp]=False
                continue
            if np.any(res[-1]<self.steadystate*0.0001): # A species has died of.
                self.validExperiment[exp]=False
                continue
            
        
    def constructBeta(self):
        tempBeta = np.zeros(shape=(self.numberOfExperiments,self.numberSpecies+1,self.numberSpecies))
        # growthrate:
        tempBeta[:,0,:] = self.growth
        tempBeta[:,1:,:] = self.interactionMatrix
        self.beta = tempBeta
                     
    def isStable(self,matrix, steady):
        J = np.diag(steady).dot(matrix.T) # Jacobian, # Transpose! 
        if np.any(np.real(np.linalg.eigvals(J)) > 0):
            return False
        else:
            return True
            
    def pertubation(self,isInitial=False):
        if isInitial:
            return self.steadystate + self.pertuPar["strenght"]*np.random.randn(self.numberOfExperiments,self.numberSpecies)
        else:
            number = int(self.pertuPar["period"]/self.timestep)
            if self.i%number==0:
                return self.pertuPar["strenght"]*np.random.randn(self.numberSpecies)
            else:
                return np.zeros(self.numberSpecies)
            
    def GLV(self, currentState):
        if self.parametersAreGen == False:
            print("Error: model parameters have not been generated")
            return
        return currentState*(self.growth[self.e] + np.dot(currentState,self.interactionMatrix[self.e]))
        
        