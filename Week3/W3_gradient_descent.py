
import numpy as np
import time

def gradient_descent(Start,gradientmodel,max_itt=100,precision=0.01,alpha=0.1):
    Theta = Start
    diff = np.Inf
    i = 0
    t0 = time.time()
    while diff>precision and i<max_itt:
        ThetaOld = Theta
        Gradient = gradientmodel(Theta)
        Theta = Theta - alpha*Gradient
        diff = np.max(np.abs(ThetaOld-Theta))
        i += 1
    # Print the ellapsed time
    t1 = time.time()
    if(round(t1-t0)==0):# Print ms  
        print("Ellapsed time = {} ms".format(round( (t1-t0)*1e3,2) ))
    else: # Print seconds
        print("Ellapsed time = {} s".format(round( (t1-t0),2) ))

    # Based on how the while loop ends print the result. 
    if i == max_itt:
        print("max number of itteration {} was reached where last diff={}".format(max_itt,diff))
    else:
        print("Desired precision {} was reached with diff = {} in {} iterations".format(precision,diff,i))
    return Theta


def DescSpartan(Start,gradientmodel,max_itt=100,alpha=0.1,verbose=1):
    Theta = Start
    i = 0
    t0 = time.time()
    while i<max_itt:
        Theta = Theta - alpha*gradientmodel(Theta)
        i += 1
    # Print the ellapsed time
    t1 = time.time()
    if verbose:
        if(round(t1-t0)==0):# Print ms  
            print("Ellapsed time = {} ms".format(round( (t1-t0)*1e3,2) ))
        else: # Print seconds
            print("Ellapsed time = {} s".format(round( (t1-t0),2) ))
    
    return Theta
    
