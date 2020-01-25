import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt


# Function that returns whether a fixed point is stable
def is_stable(interactionmatrix, SeconcOrderMatrix, steadystate):
    # compute jacobian
    Nspecies = interactionmatrix.shape[0]
    hor_steady = steadystate.reshape([1,Nspecies])
    J = hor_steady*interactionmatrix + 2*hor_steady*SeconcOrderMatrix*steadystate

    if np.any(np.real(np.linalg.eigvals(J)) > 0):
        return False
    else:
        return True

def is_stable_simple(interactionmatrix, SeconcOrderMatrix): # Steady state is all one. 
    J = interactionmatrix + 2*SeconcOrderMatrix
    if np.any(np.real(np.linalg.eigvals(J)) > 0):
        return False
    else:
        return True


def main(seed=None,noise=0.05, s1=0.05, s2=0.01, selfint=None, g_m = 1, g_s = 0.01, pertubation = 0.1, period_pertu = 20, plot=True):

    if seed == None:
        np.random.seed(int(time.time()))
    else:
        np.random.seed(seed)

    # number of species
    Nspecies = 5

    # Fix the steady state. In this way we can make sure that none of the species has a negative steady state.
    steadystate = np.ones([Nspecies, 1])
    # emmegration rate
    immigration_rate = np.repeat(0.1, Nspecies).reshape([Nspecies, 1])

    #sigma = 0.05 # strength of the interactions

    #------------------
    # Generate parameter according to the given strenght. 
    stable = False
    counter = 0
    while not stable:
        # First order matrix
        if s1 == 0:
            Interaction_Matrix = np.zeros([Nspecies, Nspecies])
        else:
            Interaction_Matrix = np.random.normal(0, s1, [Nspecies, Nspecies])

        # Self interaction
        if selfint == None:
            np.fill_diagonal(Interaction_Matrix, -1) # self-interaction is -1, this is a convention. Rethink this assumption I would denial this assumption !!!
        elif selfint == "Keystone":
            np.fill_diagonal(Interaction_Matrix, np.random.uniform(-1.9/steadystate, -0.1/steadystate, Nspecies))# Here we take the same approuch as the keystone paper. 
        elif type(selfint)==float:
            np.fill_diagonal(Interaction_Matrix, np.random.uniform(-1.9*selfint, -0.1*selfint, Nspecies))# aka we want it to be small. 

        # Second order matrix
        if s2 == 0:
            SecondOrder_matrix = np.zeros([Nspecies, Nspecies])
        else:
            SecondOrder_matrix = np.random.normal(0, s2, [Nspecies, Nspecies])
        np.fill_diagonal(SecondOrder_matrix, 0) 

        # randomly generate growth rate. 
        growthrate = np.random.normal(g_m, g_s, [Nspecies, 1])

        # Define the allee factory by the steady state
        allee_factor = immigration_rate/growthrate - Interaction_Matrix.dot(steadystate) - SecondOrder_matrix.dot(np.power(steadystate,2))

        # Check stability
        if np.all(steadystate==1):
            stable = is_stable_simple(Interaction_Matrix, SecondOrder_matrix)
        else:
            stable = is_stable(Interaction_Matrix, SecondOrder_matrix, steadystate)
        counter += 1

    #------------------

    # Initial condition is perturbed steady state. 
    initial_condition = steadystate * np.random.normal(1, pertubation, [Nspecies, 1])

    params = {}
    params['Interaction_Matrix'] = Interaction_Matrix
    params['SecondOrder_matrix'] = SecondOrder_matrix
    params['growthrate'] = growthrate
    params['allee_factor'] = allee_factor
    params['immigration_rate'] = immigration_rate
    params['initial_condition'] = initial_condition
    params['steadystate'] = steadystate
    params['Attempts'] = counter

    # Generate a timeseries
    res = run_timeseries_noise(params,noise=noise,  period_pertu= period_pertu)
    ts = res['Species_abundance']

    # Plot the timeseries
    if plot:
        # Print some extra things
        print("It took {} attempts to find a stable state".format(counter))

        fig = plt.figure(constrained_layout=True,dpi=100)
        ax = fig.add_subplot(111)

        for i in range(1, Nspecies+1):
            ax.plot(ts['time'], ts['species_%d' % i], label='species %d' % i)

        ax.legend()
        plt.show()

    params['timeseries'] = ts 
    params['model'] = res['model']
    params['noise'] = res['noise']
    return params


def run_timeseries_noise(params, noise=0.05, period_pertu= 20 ,dt=0.01, T=100, tskip = 0):
    
    # Set seed for random number generator
    r, a, B, C, d, initcond = params['growthrate'], params['allee_factor'], params['Interaction_Matrix'], params['SecondOrder_matrix'], params['immigration_rate'], params['initial_condition']
    Nspecies = len(r)
    x_ts = np.copy(initcond) # The whole timeseries
    x = np.copy(initcond) # The current state
    initial_pertu = initcond-1

    dx_ts = np.array([[],[],[],[],[]]) # create empty (5,1) that will be filled up. 
    dn_ts = np.array([[],[],[],[],[]])
    # Integrate ODEs according to model and noise
    for i in range(1, int(T / dt)):
    	# LANGEVIN_LINEAR
        dx = ( x *( r *( a + B.dot(x) + C.dot(x*x))- d) ) * dt # Euler method
        dn = noise * x * np.sqrt(dt) * np.random.normal(0, 1, x.shape) # actually the sqrt of the time results in a larger noise effect !!! per time step. then if we let it be linear.
        pertu = 0 if (i*dt)%period_pertu!=0 and i!=0 else np.random.normal(1, 0.1, [5, 1])
        x += dx + dn + pertu

        if i % (tskip + 1) == 0:
            dx_ts = np.hstack((dx_ts, dx))
            dn_ts = np.hstack((dn_ts, dn))
            x_ts = np.hstack((x_ts, x))

    
    # return timeseries if ts = True, else return only endpoint
    x_ts = np.vstack((dt * (tskip+1)*np.arange(len(x_ts[0]))[np.newaxis, :], x_ts))
    x_ts = pd.DataFrame(x_ts.T, columns=['time'] + ['species_%d' % i for i in range(1,Nspecies+1)])
    result = {} 
    result['Species_abundance'] = x_ts
    result['model'] = dx_ts
    result['noise'] = dn_ts
    return result


if __name__ == "__main__":
    main()