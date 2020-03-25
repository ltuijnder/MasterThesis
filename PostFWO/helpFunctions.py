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