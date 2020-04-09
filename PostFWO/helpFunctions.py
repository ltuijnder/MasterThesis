import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

def total_error(Theta1,Theta2):
    error=np.abs(Theta1-Theta2)
    return(np.sum(error))

    
def plotM(Matrix,title,mode="Diff"):
    fig = plt.figure(constrained_layout=True)
    setColorBar= True
    ax = fig.add_subplot(111)
    if mode=="Diff":
        cax = ax.matshow(Matrix, cmap= "jet")
    elif mode=="PosNeg":
        vmin = np.min(Matrix)
        vmax = np.max(Matrix)
        dnorm = matplotlib.colors.DivergingNorm(vmin=vmin,vcenter=0,vmax=vmax)
        cax = ax.matshow(Matrix, cmap= "bwr",norm=dnorm)
    elif mode=="Sigma":
        vmin = 2 # Anything less then 2 sigma is basically not signifcant. 
        vmax = 6 # Everything above 6 sigma is extremly significant.
        vcenter = 3 # 3 sigma should be the threshold.
        dnorm = matplotlib.colors.DivergingNorm(vmin=vmin,vcenter=vcenter,vmax=vmax)
        # convert infinit cases to 10 sigma such that the white is not given
        copyM = np.copy(Matrix)# Make first an hard copy such that we do not change anything to the original
        copyM[np.isfinite(copyM)==False]=10 # Replace it with 10 sigma. Which is redicoulisly small
        
        cax = ax.matshow(copyM, cmap= "jet",norm=dnorm)
        cbar = fig.colorbar(cax)
        cbar.set_label("Sigma",fontsize = 15)
        cbar.ax.set_yticklabels(['< 2.0', '2.5', '3.0','3.5','4.0','4.5','5.0','5.5','>6.0'])
        
        setColorBar = False # Set colorbar since we have already set it.
        
    plt.title(title,fontsize=18,pad=28)
    ax.set_xlabel('Column',fontsize=15)    
    ax.xaxis.set_label_position('top') 
    ax.set_ylabel('Row',fontsize=15)
    if setColorBar:
        fig.colorbar(cax)
    plt.show()
    
def plot3DY(Y,X,Exp = 0,Ynumber = 0):
    fig = plt.figure(figsize = (5,5))
    ax = fig.add_subplot(111,projection='3d')
    
    YData = Y[Exp,:,Ynumber]
    testXSpecies1 =  X[Exp,:,1]
    testXSpecies2 =  X[Exp,:,2]
    ax.plot(testXSpecies1,testXSpecies2,YData,"*", alpha=0.1)
    plt.xlabel("Species1")
    plt.ylabel("Species2")
    
    
def plot2DScatter(Y,X,xaxis="1",yaxis="2",Exp = 0,Ynumber = 0):
    fig = plt.figure(figsize = (5,5))
    ax = fig.add_subplot(111)
    
    YData = Y[Exp,:,Ynumber]
    testXSpecies1 =  X[Exp,:,1]
    testXSpecies2 =  X[Exp,:,2]
    
    if xaxis == "1":
        X_ = testXSpecies1
    elif xaxis == "2":
        X_ = testXSpecies2
    else:
        pass
    
    if yaxis == "1":
        Y_ = testXSpecies1
    elif yaxis == "2":
        Y_ = testXSpecies2
    else:
        Y_ = YData
        
        
    ax.plot(X_,Y_,"*",alpha=0.5)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    

def offDiagonalGLVBEst(GLVBEst):
    matrix = GLVBEst[1:]
    n = matrix.shape[0]
    non_diag = (np.ones(shape=(n,n)) - np.identity(n)).astype(bool)
    return matrix[non_diag]
    
##########
### Other misolaniance plots that are just often used.
##########
    
def plotSubSampMatrix(variable = "percent",section="I", stepSamples = (1,2,3), Self = False,pathToStorage="DataStorage/SubSampleBis/"):
    expoNoise = np.arange(-5,-0,0.5)
    noises = np.power(10,expoNoise)
    expoInteraction = np.arange(-4.5,0.5,0.5)
    interactions = np.power(10,expoInteraction)
    pertubations = np.array([10000,20,2])
    subSampleSteps = np.array([1,2,3,4,5,7,9,11,13,15,20,25,30,35,40,60,80,100,120,140])
    # Load wanted data
    if variable == "numberOfGoodExp":
        path = "numberOfGoodExp.npy"
    else:
        path = "MEDIAN"+variable+section+".npy"
    if Self:
        path = "Self_" +  path
    medianData = np.load(pathToStorage+path)
    
    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(15,15))
    fig.suptitle(f"Median {variable}{section} subsampled", fontsize=22)
        
    #loop over the axes
    
    for row, axs_row  in enumerate(axs):
        
        for column, ax in enumerate(axs_row):
            # create labels
            if row==0:# The top row
                ax.set_title(f"stepSamples = {stepSamples[column]}",fontsize=15)
            if column==0: # The right side
                ax.set_ylabel(f"period = {pertubations[row]}",fontsize=15)
            
            l = np.where(subSampleSteps==stepSamples[column])[0][0] # get back the correct index
            
            # Plot the data
            if variable=="percent":
                im = ax.matshow(medianData[row,:,:,l],cmap="jet",vmin=0, vmax=1)
            elif variable == "numberOfGoodExp":
                im = ax.matshow(medianData[row]/50,cmap="jet",vmin=0, vmax=1)
            else:
                im = ax.matshow(medianData[row,:,:,l],cmap="jet",vmin=0, vmax=6)
            
    
    # Tick lables for some weird reason have to be done this way :/ 
    rightSideAxes = (axs[0][0],axs[1][0],axs[2][0])
    lowerSideAxes = (axs[2][0],axs[2][1],axs[2][2])
    topSideAxes = (axs[0][0],axs[0][1],axs[0][2])
    for ax in rightSideAxes:
        ax.set_yticks(np.arange(len(expoNoise)))
        ax.set_yticklabels(labels=expoNoise)
    for ax in lowerSideAxes:
        ax.set_xticks(np.arange(len(expoInteraction)))
        ax.set_xticklabels(labels=expoInteraction)
        ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=False)
    for ax in topSideAxes:
        pass
        #ax.set_xticks(np.arange(7))
        #ax.set_xticklabels(labels=interStength)
        #ax.tick_params(axis="x", bottom=True, top=True, labelbottom=False, labeltop=True)
    # Add color bar
    cax = fig.add_axes([0.9, 0.3, 0.03, 0.4])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(variable,fontsize=15)
    # Add x and Y label
    fig.text(0.5,0.1, "Log. Interaction strenght", ha="center", va="center",fontsize=17)
    fig.text(0.07,0.5, "Log. noise strenght", ha="center", va="center", rotation=90,fontsize=17)  
    # adjust space between pltos
    
    plt.subplots_adjust( wspace=0, hspace=0.02)
        