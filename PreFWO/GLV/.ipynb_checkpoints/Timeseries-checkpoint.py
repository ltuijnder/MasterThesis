
# Imperial code of generating the normal GLV data

import numpy as np
import time
import matplotlib.pyplot as plt


# IS NOT GENERAL!!!
def is_stable(interactionmatrix, steadystate):
	J = interactionmatrix # Jacobian

    if np.any(np.real(np.linalg.eigvals(J)) > 0):
        return False
    else:
        return True


def main(seed=None, noise=0.05, sigma=0.05, selfint=None, g_m = 1, g_s = 0.01, pertubation = 0.1, period_pertu = 20, plot=True):
	if seed == None:
        np.random.seed(int(time.time()))
    else:
        np.random.seed(seed)

    # number of species
    Nspecies = 5

    # Fix the steady state. In this way we can make sure that none of the species has a negative steady state.
    steadystate = np.ones([Nspecies, 1])

    #sigma = 0.05 # strength of the interactions

    # Look for parameters such that the system is stable.
    stable = False
    while not stable:
        # interaction
        if sigma == 0:
            interaction_matrix = np.zeros([Nspecies, Nspecies])
        else:
            interaction_matrix = np.random.normal(0, sigma, [Nspecies, Nspecies])
        np.fill_diagonal(interaction_matrix, -1) # self-interaction is -1, this is a convention

        # no immigration
        immigration_rate = np.zeros([Nspecies, 1])

        # growthrates determined by the steady state
        growthrate = - interaction_matrix.dot(steadystate).reshape([Nspecies, 1])

        stable = is_stable(interaction_matrix, steadystate, growthrate)

    # Initial condition is perturbed steady state. 
    initial_condition = steadystate * np.random.normal(1, 0.1, [Nspecies, 1])

    params = {}
    params['interaction_matrix'] = interaction_matrix
    params['immigration_rate'] = immigration_rate
    params['growthrate'] = growthrate
    params['initial_condition'] = initial_condition

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
	omega, mu, g, initcond = params['interaction_matrix'], params['immigration_rate'], params['growthrate'], params['initial_condition']
	Nspecies = len(omega) # number of species
    x_ts = np.copy(initcond) # The whole timeseries
    x = np.copy(initcond) # The current state
    initial_pertu = initcond-1

    dx_ts = np.array([[],[],[],[],[]]) # create empty (5,1) that will be filled up. 
    dn_ts = np.array([[],[],[],[],[]])

    # Integrate ODEs according to model and noise
    for i in range(1, int(T / dt)):
    	# LANGEVIN_LINEAR
    	dx = (omega.dot(x) * x + mu + g * x) * dt
        dn = noise * x * np.sqrt(dt) * np.random.normal(0, 1, x.shape)
        x += dx + dn
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