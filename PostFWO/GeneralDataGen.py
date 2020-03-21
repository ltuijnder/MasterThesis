import numpy as np
import matplotlib.pyplot as plt

seedList1 = [  785, 38169, 44682, 15570, 13274, 44387, 28742, 34599, 39125, 18973]
seedList2 = [13367,  5901,  8258,  2184, 39489, 11901, 21542, 25640, 12128, 11222]

class Data:
    def __init__(self, inputDim, outputDim, numberOfObservations=500, numberOfExperiments=2, typeData = "simpleLinear"):
        # Data specefic
        self.numberOfExperiment = numberOfExperiments
        self.numberOfObservations = numberOfObservations
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.reset()# create the empty cells
        
        # object specific
        self.typeData = typeData
        self.seeds = seedList1[:self.numberOfExperiment]
        self.isGenerated = False
        
        # sampling noise methods
        self.noiseType = "gaussian"
        self.noisePar = {"noiseStd":np.repeat(1,self.outputDim),"noiseMean":np.repeat(0,self.outputDim)}
        
        # sampling x methods
        self.sampleXType = "uniform"
        self.sampleXPar = {"low":np.repeat(0,self.inputDim),"high":np.repeat(1,self.inputDim)} # Parameters belong to the generation of X
        
        # Parameters for simpleLinear
        if self.typeData == "simpleLinear":
            self.coefMatrix = np.arange( (self.inputDim+1) * self.outputDim).reshape((self.inputDim+1),self.outputDim)
    
    def reset(self):
        self.isGenerated = False
        # For now convert the x and y to lists... this should be numpy but hey... If I have time I will converge it. 
        self.x = []
        self.y = []
        self.modelContribution = []
        self.noiseContribution = []
    
    def setNoise(self, noiseType, noisePar):
        # First check if everything is set correctly
        isCorrectInput = False
        if noiseType == "gaussian":
            try:
                noiseStd = noisePar["noiseStd"]
                noiseMean = noisePar["noiseMean"]
                if len(noiseStd)!=self.outputDim or len(noiseMean)!=self.outputDim:
                    print(f"Invalid outputDim lenght of {len(low),len(high)}. The correct outputDim is {self.outputDim}")
                    return
            except:
                print("noisePar are not correct for noiseType 'gaussian'")
                return
            isCorrectInput = True
        if isCorrectInput:
            self.noiseType = noiseType
            self.noisePar = noisePar
        else:
            print("Invalid noiseType")
    
    def setSampleX(self, sampleXType, sampleXPar):
        # first check if everything is set correctly
        isCorrectInput = False
        if sampleXType == "uniform":
            try:
                low = sampleXPar["low"]
                high = sampleXPar["high"]
                if len(low)!=self.inputDim or len(high)!=self.inputDim:
                    print(f"Invalid inputDim lenght of low,high={len(low),len(high)}. The correct inputDim is {self.inputDim}")
                    return
            except:
                print("sampleXPar are not correct for sampleXType 'uniform'")
                return
            isCorrectInput = True
        elif sampleXType == "gaussian":
            try:
                mean = sampleXPar["mean"]
                sigma = sampleXPar["sigma"]
                if len(mean)!=self.inputDim or len(sigma)!=self.inputDim:
                    print(f"Invalid inputDim lenght of mean,sigma={len(mean),len(sigma)}. The correct inputDim is {self.inputDim}")
                    return
            except:
                print("sampleXPar are not correct for sampleXType 'gaussian'")
                return 
            isCorrectInput = True
            
        if isCorrectInput:
            self.sampleXType = sampleXType
            self.sampleXPar = sampleXPar
        else:
            print("Invalid sampleXType")
    
    # Only set the parameters
    
    def setSampleXLowHigh(self,low,high):
        if self.sampleXType == "uniform":
            if len(low)!=self.inputDim or len(high)!=self.inputDim:
                    print(f"Invalid inputDim lenght of low,high={len(low),len(high)}. The correct inputDim is {self.inputDim}")
                    return
            self.sampleXPar["low"]=low
            self.sampleXPar["high"]=high
        else:
            print(f"sampleXType={self.sampleXType} and not 'uniform'")
            
    def setSampleXMeanStd(self,mean,std):
        if self.sampleXType == "gaussian":
            if len(mean)!=self.inputDim or len(sigma)!=self.inputDim:
                    print(f"Invalid inputDim lenght of mean,sigma={len(mean),len(sigma)}. The correct inputDim is {self.inputDim}")
                    return
            self.sampleXPar["mean"]=low
            self.sampleXPar["sigma"]=high
        else:
            print(f"sampleXType={self.sampleXType} and not 'gaussian'")
            
    def setNoiseStd(self,std):
        if self.noiseType=="gaussian":
            if len(std)!=self.outputDim:
                print(f"invalid outputDim lenght of std={len(std)}. The correct output dimension is {self.outputDim})")
                return
            self.noisePar["noiseStd"]=std
        else:
            print(f"noiseType={self.noiseType} and not 'gaussian'")
                    
    
    def setModelParameters(self,parameters):
        # This function is reimplemented in inherented classes! 
        if len(parameters)!=self.inputDim+1:
            print(f"Wrong inputDim input matrix. Expected array of length {self.inputDim+1}")
        else:
            self.coefMatrix = parameters
    
    def generate(self):
        """Generate random regression data
            By default it will take coefMatrix and rangeData of the class.
            When coefMatrix and/or rangeData are speciefied, it will take those instead and overwrite the original.
        """
        # Reset the data
        self.reset() # Everytime we generate we reset. 
        
        for numberExperiment in range(self.numberOfExperiment):
            # Set seed for the experiment
            np.random.seed(self.seeds[numberExperiment])
            
            # arrays that will keep the result of the observations
            xArray = np.zeros(self.inputDim).reshape(1,self.inputDim)
            yArray = np.zeros(self.outputDim).reshape(1,self.outputDim) # experimental
            modelArray = np.zeros(self.outputDim).reshape(1,self.outputDim) # model contribution
            noiseArray = np.zeros(self.outputDim).reshape(1,self.outputDim)
            
            # Make the observations of the experiment
            for observation in range(self.numberOfObservations):
                # Compute observations
                (xValue,y,yModel,noise) = self.makeObservation(xArray,yArray)
                
                # Save individual contributions
                xArray = np.append(xArray,xValue.reshape(1,self.inputDim),axis=0)
                modelArray = np.append(modelArray,yModel.reshape(1,self.outputDim),axis=0)
                noiseArray = np.append(noiseArray,noise.reshape(1,self.outputDim),axis=0)
                yArray = np.append(yArray,y.reshape(1,self.outputDim),axis=0)
            
            # Save and remove the first zero dummy line
            # Important the code below assumes that on reset the self.x,y are set to lists. 
            self.x.append(xArray[1:])
            self.y.append(yArray[1:])
            self.modelContribution.append(modelArray[1:])
            self.noiseContribution.append(noiseArray[1:])
            
        # Convert to numpy, since it is easier to work with
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.modelContribution = np.array(self.modelContribution)
        self.noiseContribution =  np.array(self.noiseContribution)
        
        self.isGenerated = True
    
    def makeObservation(self,xArray,yArray):
        """ Return the next observation of the experiment
            inputs all the observations that were made up until now
            returns (x,y,model,noise)
        """
        
        # For simple functional the current information is not needed
        xValue = np.array(self.sampleX())
        xMatrix = np.append(1,xValue)    
        yModel = np.dot(xMatrix,self.coefMatrix)
        noise = self.sampleNoise()
        return (xValue,yModel+noise,yModel,noise)
    
    def sampleX(self):
        if self.sampleXType == "uniform":
            return np.random.uniform(self.sampleXPar["low"],self.sampleXPar["high"])
        elif self.sampleXType == "gaussian":
            return np.random.normal(self.sampleXPar["mean"],self.sampleXPar["sigma"]) 
    
    def sampleNoise(self):
        if self.noiseType == "gaussian":
            return np.random.normal(self.noisePar["noiseMean"], self.noisePar["noiseStd"])
        
    def getData(self,i=None):
        """return data of experiment i. 
            If i == None (which is default) return the whole tuple
        """
        if i is None:
            return self.listOfData
        return self.listOfData[i]
    
    def plotData(self, inputDimNumber=0, outputDimNumber=0, experimentNumber=None):
        if inputDimNumber>=self.inputDim:
            print("inputDimNumber is to high")
        for i in range(self.numberOfExperiment):
            if experimentNumber is not None: # Go further
                if i!=experimentNumber:
                    continue
            
            x = self.x[i][:,inputDimNumber].flatten()
            y = self.y[i][:,outputDimNumber].flatten()
            plt.scatter(x,y,
                     label=f"Exp. {i}")
            
        plt.xlabel(f"inputDim {inputDimNumber}")
        plt.ylabel(f"outputDim {outputDimNumber}")
        plt.title(f"Plot x{inputDimNumber} vs y{outputDimNumber}")
        plt.legend()
    