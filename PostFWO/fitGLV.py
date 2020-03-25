import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from helpFunctions import * # Contains the plotting stuff.


class fitGLV:
    def __init__(self, data, typeInput = "TS_GLV"):
        self.typeInput = typeInput
        
        # Incase of normal TS input 
        if self.typeInput == "TS_GLV" and data.isGenerated: # Python is lazy in checking!
            self.TS = data
            self.trueMat =  self.TS.beta
        elif self.typeInput == "Data":
            self.data = data
            numberOfExperiments, numberOfPoints, numberOfSpecies = self.data.x.shape
            self.trueMat = np.repeat(self.data.coefMatrix.reshape(
                1,numberOfSpecies+1,numberOfSpecies),numberOfExperiments,axis=0)
        else:
            print("Error: The timeseries needs to be generated.")
        # compute the relevant y for the model. This can be overwritten for different models
        self.computeY() # output
        self.computeX() # input
        
        # Compute number of parameters, to incorporate the look else where effect
        nSpecies = self.Y.shape[-1]# number of columns
        self.numberOfParameters = nSpecies*(nSpecies+1)
        
        # Flags
        self.isFitted = False
        self.varIsEstimated = False
        self.pValueIsComputed = False
    
    def fitLinear(self):# General fit function. This creates all the wanted variables.
        # Construct data matrix X.
        XT = self.X.transpose(0,2,1)# Assume 3D shape. But can be easily done more general. 
        
        # Now compute the estimated beta. "@" is the matrix multiplication and can handle stacked matrixes like here!
        self.BEst = np.linalg.inv(XT@self.X)@XT@self.Y # BEst = beta estimate
        
        self.isFitted = True
    
    def computeVarBEst(self):
        # First we need to compute the predicted Y. For this check if the parameters have been estimated else do that first
        if not self.isFitted: # Automatically compute Beta estimate if it was not computed before
            self.fitLinear()
        XT = self.X.transpose(0,2,1)# Assume 3D shape. But can be easily done more general. 
        
        YPredict = self.X@self.BEst
        
        # Next we compute the sample variance of the data
        diffSquare = np.power(self.Y-YPredict,2)
        normFactor = np.power(float(YPredict.shape[-2] - self.BEst.shape[0]),-1) # We estimate var per species, hence we need number of degree of one species which is shape[0]
        self.varEst = normFactor * np.sum(diffSquare,axis = -2) # Sum over the column aka sum over one species
        
        # Next we compute the variance of the parameters
        # Important is that var(B_i)=(XT@X)*varEst_i, 
        # thus for species i which is a column i in BEst has now a matrix. Before we kept this matrix. Here we just keep the diag and convert it to column.
        
        varOneSpecies = np.diagonal(np.linalg.inv(XT@self.X),axis1=-2,axis2=-1)
        lenDiag = varOneSpecies.shape[-1]
        # Take the diagonal which is now a row vector and make it into a column and copy the column multiple times
        copied = np.repeat(varOneSpecies.reshape(varOneSpecies.shape + (1,)), self.varEst.shape[-1], axis=-1)
        # The estimated varBEst is now each column multiplied by its respective sigma. For this we first need to broadcast the sigmas into the correct shape
        self.varBEst = copied * np.repeat( self.varEst.reshape(self.varEst.shape[-2],1,self.varEst.shape[-1]), lenDiag, axis=-2) 
        # What is happening is easier to see with the following test examples
        # varEst = np.array([[1,2,3],[4,5,6]]) # where axis =0 # number of experiments, axis=1 number of species
        # np.repeat(test.reshape(2,1,3),2,axis=-2) # we then add an axis in the middle and copy that number of parameters needed for 1 species = lenDiag
        self.varIsEstimated = True
        
        
    def computeNullHypo(self):
        # first compute the estimated variance if has not been done yet
        if not self.varIsEstimated:
            self.computeVarBEst()
        
        # First compute the Z matrix of which the elements follow a student-t distrubution of degree DF.
        Z = self.BEst/np.sqrt(self.varBEst)
        DF = self.Y.shape[-2]-self.BEst.shape[-2]# self.Y.shape[-2] = number of effective used points used. Without pertubation = number of timepoints.

        # Compute p-Value NullHypo
        self.pNull = 2*(1 - stats.t.cdf(np.abs(Z),df=DF))
        # important is that we used the absolute value in order to do two sided
        # Also compute the p-value interms of sigma, for this use the inverse cumulative of normal = ppf
        
        # Incorporation of the look else where effect. Whe make the result less significant
        # By multiplying with the number of parameters.
        #self.pNull*=self.numberOfParameters
        self.pNull[self.pNull>1] = 1 # When multiplied by number parameters. Prob. cannot go above 1. 
        self.pNullSigma = np.abs(stats.norm.ppf(self.pNull/2))
        self.pValueIsComputed = True
        
        self.nullSummary = computeSummary(self.pNull, self.pNullSigma, self.trueMat)
        
    def hypo(self,betaHypo,ExpNum=None,plot=True, plotNumb=0):
        # ExpNum: if betaHypo is just one experiment matrix. then it should be said to which exp it should be compared.
        if not self.varIsEstimated:
            self.computeVarBEst()
        
        if ExpNum is None:
            Z = (self.BEst-betaHypo)/np.sqrt(self.varBEst)
        else:
            Z = (self.BEst[ExpNum]-betaHypo)/np.sqrt(self.varBEst[ExpNum])
        
        DF = self.Y.shape[-2]-self.BEst.shape[-2]# self.Y.shape[-2] = number of effective used points used. Without pertubation = number of timepoints.
        p = 2*(1 - stats.t.cdf(np.abs(Z),df=DF))
        #p *= self.numberOfParameters # look else where effect
        p[p>1]=1 # When multiplied by number of parameters. Prob. cannot go above 1. 
        pSigma = np.abs(stats.norm.ppf(p/2))
        
        if plot:
            if ExpNum is None:
                plotM(pSigma[plotNumb],f"Hypo. P-value exp. {plotNumb}",mode="Sigma")
            else:
                plotM(pSigma,f"Hypo. P-value exp. {ExpNum}",mode="Sigma")
        
        summary = computeSummary(p, pSigma, self.trueMat, withModified = False)
        return (p, pSigma, computeSummary)
            
    def plotNullHypo(self,number=0):
        if not self.pValueIsComputed:
            self.computeNullHypo()
        # This function requires that the function "plotM" from helpFunctions is imported
        plotM(self.pNullSigma[number],f"Null Hypo. P-value exp. {number}",mode="Sigma")      
    
    def computeY(self): # Since this class is called GLV. but can be overwritten in child classes
        if self.typeInput == "TS_GLV":
            dln = np.diff(np.log(self.TS.result),axis=-2) # Axis =-1 is species, -2 = temporal, -3 and higher is experiment and batch
            self.Y = dln/self.TS.timestep
            
            # Remove pertubed entries, since these lead to false Y values.
            boolHasPertu = self.TS.hasPerturbed[0]# For now it is assumed that pertubation for all exp are same. 
            self.Y =  self.Y[:,~boolHasPertu] # "~" = Not      
        elif self.typeInput == "Data":
            self.Y = self.data.y
        else:
            print("Not supported inputData")
            return
        
    def computeX(self):
        if self.typeInput == "TS_GLV":
            ones = np.ones(shape=(self.TS.numberOfExperiments, self.TS.numberOfPoints, 1)) # Watch out not pertubation dependent. 
            FullX = np.append(ones,self.TS.result, axis = -1)
            # However we need to remove the last element since our output Y is calculated based on output on a difference which can not be computed for the last element. 
            self.X = np.delete(FullX, -1, axis = -2)
            
            # Remove pertubed entries, since these lead to false Y values.
            boolHasPertu = self.TS.hasPerturbed[0]# For now it is assumed that pertubation for all exp are same. 
            self.X =  self.X[:,~boolHasPertu] # "~" = Not
        elif self.typeInput == "Data":
            numberOfExperiments, numberOfPoints, numberOfSpecies = self.data.x.shape
            ones = np.ones(shape=(numberOfExperiments, numberOfPoints, 1)) # Watch out not pertubation dependent.
            self.X = np.append(ones,self.data.x, axis = -1)
            # Here we do not remove the last layer since we did not compute any difference with the Y
        else:
            pass
        
        
#############################
### HELP FUNCTION ###########
#############################

def computeSummary(p, sigma, trueMatrix, withModified = True):
    # e = number experiments, n = number species
    # p, sigma = 3 dimensional matrixes.
    # trueMatrix = true 3 dimensional coeffiecient matrix
    e, n1, n = trueMatrix.shape # e = number experiments, n = number species, n1= n+1
    # Save the summarys in a dictionary
    Summary = {}
    
    #############################################################
    # first we need to make check that they are non valid entries
    indexInvalidExp = []
    for i in range(e):
        isNan = np.any(np.isnan(p[i]))
        if isNan:
            indexInvalidExp.append(i) # Store the index of the invalid experiment
            p[i] = np.ones((n1,n)) # replace p and sigma with default values
            sigma[i] = np.zeros((n1,n)) # Such that they do not cause problems down the line.
    Summary["indexInvalidExp"] =  np.array(indexInvalidExp)
    ##############################################################
    
    # Next get the element class specif sigma's.
    sigmaG = sigma[:,0,:] # sigma Growth rate.
    sigmaS = np.copy(np.diagonal(sigma, offset=-1, axis1=-2, axis2=-1))# sigma Self interaction
        
    # to get the non-diagonal elements, slice the matrix with another matrix of which the off-diagonal are true. Achieve this by putting the diagonal elements = NaN and then use the test np.isnan
    withOutG = np.copy(sigma[:,1:,:]) # Shave of the Growth rate
    indexAxis0 = np.repeat(np.arange(e),n)# Example n=3, e=2 [0,0,0,1,1,1]
    indexAxis1And2 = np.tile(np.arange(n),e)# Example  n=3, e=2 [0,1,2,0,1,2]
    
    areAlreadyNan = np.isnan(withOutG)# It is possible for there to be Nan.
    withOutG[(indexAxis0 , indexAxis1And2 , indexAxis1And2)] = np.nan # Put the diagonal of the stacked matrix = Nan
    sigmaI = withOutG[~np.isnan(withOutG)].reshape(e,-1) # sigma Interaction.
    ###############
    # Now compute the wanted statistics per group and a weighted average

    signThreshold = 3 # significance threshold.
    w = np.array([15,10,75]) # [G,S,I]
    totW = np.sum(w)
    
    # Replace potential infinities with maxSigma,
    maxSigma =  10 # Potential bias!! 
    sigmaG[~np.isfinite(sigmaG)] = maxSigma
    sigmaS[~np.isfinite(sigmaS)] = maxSigma
    sigmaI[~np.isfinite(sigmaI)] = maxSigma
    
    # Percentage significant.
    Summary["percentG"] = np.sum(sigmaG>signThreshold,axis=-1)/n # Sum per experiment
    Summary["percentS"] = np.sum(sigmaS>signThreshold,axis=-1)/n
    Summary["percentI"] = np.sum(sigmaI>signThreshold,axis=-1)/(n*(n-1))
    Summary["percentWAvg"] = np.sum(np.array([w[0]*Summary["percentG"],
                                              w[1]*Summary["percentS"],
                                              w[2]*Summary["percentI"]]),axis=0)/totW
    
    # Median sigma
    Summary["medianG"] = np.median(sigmaG ,axis = -1)
    Summary["medianS"] = np.median(sigmaS ,axis = -1)
    Summary["medianI"] = np.median(sigmaI ,axis = -1)
    Summary["medianWAvg"] = np.sum(np.array([w[0]*Summary["medianG"],
                                             w[1]*Summary["medianS"],
                                             w[2]*Summary["medianI"]]),axis=0)/totW
    
    # Mean sigma
    Summary["avgG"] = np.mean(sigmaG ,axis = -1)
    Summary["avgS"] = np.mean(sigmaS ,axis = -1)
    Summary["avgI"] = np.mean(sigmaI ,axis = -1)
    Summary["avgWAvg"] = np.sum(np.array([w[0]*Summary["avgG"],
                                          w[1]*Summary["avgS"],
                                          w[2]*Summary["avgI"]]),axis=0)/totW
        
    if not withModified:# Then we already return else we compute further with modifier.
        return Summary
    
    ################################################################################
    # Compute modifier and then the modified sigma
    absSelfInt =  np.abs(np.diagonal(trueMatrix, offset=-1, axis1=-2, axis2=-1))# selfinteraction 
    # cast it into the shape of trueMat and compute the fraction
    fraction = np.abs(trueMatrix)/np.repeat(absSelfInt.reshape(e,1,n), n1, axis = -2)
    # compute the modifier. 
    fraction[fraction>1] = 1
    modifier = np.sqrt(fraction)
    Summary["modifier"]=modifier
    
    sigmaMod = np.abs(stats.norm.ppf(p*modifier/2)) # Mod = modified
    #################################################################################
    # Now do again the same analysis but now with the modified. 
    
    sigmaGMod = sigmaMod[:,0,:]
    sigmaSMod = np.copy(np.diagonal(sigmaMod, offset=-1, axis1=-2, axis2=-1))
    withOutGMod = np.copy(sigmaMod[:,1:,:])
    withOutGMod[(indexAxis0 , indexAxis1And2 , indexAxis1And2)] = np.nan 
    sigmaIMod = withOutGMod[~np.isnan(withOutGMod)].reshape(e,-1) 
    sigmaGMod[~np.isfinite(sigmaGMod)] = maxSigma
    sigmaSMod[~np.isfinite(sigmaSMod)] = maxSigma
    sigmaIMod[~np.isfinite(sigmaIMod)] = maxSigma
    Summary["percentGMod"] = np.sum(sigmaGMod>signThreshold,axis=-1)/n
    Summary["percentSMod"] = np.sum(sigmaSMod>signThreshold,axis=-1)/n
    Summary["percentIMod"] = np.sum(sigmaIMod>signThreshold,axis=-1)/(n*(n-1))
    Summary["percentWAvgMod"] = np.sum(np.array([w[0]*Summary["percentGMod"],
                                                 w[1]*Summary["percentSMod"],
                                                 w[2]*Summary["percentIMod"]]),axis=0)/totW
    Summary["medianGMod"] = np.median(sigmaGMod ,axis = -1)
    Summary["medianSMod"] = np.median(sigmaSMod ,axis = -1)
    Summary["medianIMod"] = np.median(sigmaIMod ,axis = -1)
    Summary["medianWAvgMod"] = np.sum(np.array([w[0]*Summary["medianGMod"],
                                                w[1]*Summary["medianSMod"],
                                                w[2]*Summary["medianIMod"]]),axis=0)/totW
    Summary["avgGMod"] = np.mean(sigmaGMod ,axis = -1)
    Summary["avgSMod"] = np.mean(sigmaSMod ,axis = -1)
    Summary["avgIMod"] = np.mean(sigmaIMod ,axis = -1)
    Summary["avgWAvgMod"] = np.sum(np.array([w[0]*Summary["avgGMod"],
                                             w[1]*Summary["avgSMod"],
                                             w[2]*Summary["avgIMod"]]),axis=0)/totW
    return Summary
    