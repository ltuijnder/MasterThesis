import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

def fY(data,all_ = 0):
    # Do a type check
    if type(data) == pd.core.frame.DataFrame:
        data = data.values
    elif type(data) != np.ndarray:
        raise TypeError("Expected type 'numpy.ndarray' or 'pandas.core.frame.DataFrame' was given '{}'".format(type(data)))
    ts = data[:,1:].T #Species abudance Y
    grow = np.ones([1,ts.shape[1]])# Growth rate Y
    Y_all = np.append(grow,ts,axis=0) # has L+1 time elements
    if all_:
        return Y_all # return all L+1 elements
    else:
        return np.delete(Y_all, -1, axis = 1) # return L elements (removing the last one)

def fF(data):
    if type(data) == pd.core.frame.DataFrame:
        data = data.values
    elif type(data) != np.ndarray:
        raise TypeError("Expected type 'numpy.ndarray' or 'pandas.core.frame.DataFrame' was given '{}'".format(type(data)))
    dt = np.diff(data[:,0])
    ts = fY(data,1)[1:,:]# Have all L+1 elements inorder to caluclate the L long F, remove the first since it is the growth rate. 
    dln = np.diff(np.log(ts))
    return dln/dt

def fD(lambda_mu,lambda_M,nspecies):
    return np.diag( np.append( lambda_mu, np.repeat(lambda_M,nspecies) ))   

def RidgeGradientCheck(Theta):
    if ('Y' or 'F' or 'D') not in globals():
        raise SystemError("Data matrixes 'Y','F' and 'D' need to be defined globaly")
    Check=np.array([Y.shape[1]!=F.shape[1], Theta.shape[0]!=F.shape[0], Theta.shape[1]!=Y.shape[0], Theta.shape[1]!=D.shape[0]])
    if any(Check):
        vs=["Y vs F", "Theta vs F", "Theta vs Y", "Theta vs D"]
        raise ValueError("Dimensions for: {}, do not match. Cannot perform matrix calculation".format(vs[Check]))
    return 2*(np.dot( np.dot(Theta,Y)-F ,Y.T) + np.dot(Theta,D))

def RidgeGrad(Theta):
    return 2*(np.dot( np.dot(Theta,Y)-F ,Y.T) + np.dot(Theta,D))

def Solution():
    return np.dot(F,np.dot(Y.T, np.linalg.inv(np.dot(Y,Y.T) - D) ))

def SolutionFYD(F,Y,D):
    return np.dot(F, np.dot(Y.T, np.linalg.inv(np.dot(Y,Y.T) - D) ))

def total_error(Theta1,Theta2):
    error=np.abs(Theta1-Theta2)
    return(np.sum(error))

    
def plotM(Matrix,title,mode="Diff"):
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111)
    if mode=="Diff":
        cax = ax.matshow(Matrix, cmap= "jet")
    elif mode=="PosNeg":
        vmin = np.min(Matrix)
        vmax = np.max(Matrix)
        dnorm = matplotlib.colors.DivergingNorm(vmin=vmin,vcenter=0,vmax=vmax)
        cax = ax.matshow(Matrix, cmap= "bwr",norm=dnorm)
    plt.title(title,fontsize=18)
    fig.colorbar(cax)
    plt.show()