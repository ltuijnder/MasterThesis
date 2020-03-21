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