import numpy as np
import time
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
# from brownian import *  # only needed for ARATO_LINEAR noise


class NOISE(Enum):
    LANGEVIN_CONSTANT = 1
    LANGEVIN_LINEAR = 2
    LANGEVIN_SQRT = 3
    RICKER_LINEAR = 4
    ARATO_LINEAR = 5
    ORNSTEIN_UHLENBECK = 6
    SQRT_MILSTEIN = 7

class MODEL(Enum):
    GLV = 1
    QSMI = 2 # quadratic species metabolite interaction

# Function that returns whether a fixed point is stable
def is_stable(interactionmatrix, steadystate, growthrate):
    J = interactionmatrix # Jacobian

    if np.any(np.real(np.linalg.eigvals(J)) > 0):
        return False
    else:
        return True

'''
Function to run timeseries for given model with noise

* params: dictionary of parameters of model
    if model = GLV:
        * interaction_matrix: [N,N] np.array with N the number of species
        * immigration_rate: [N,1] np.array
        * growthrate: [N,1] np.array
        * initial_condition: [N,1] np.array
* noise: float, strength of the noise
* model: MODEL, model GLV (generalized Lotka Volterra) or QSMI (quadratic species metabolite interaction)
* noise_implementation: NOISE
* dt: float, fundamental timestep
* T: float, end time for integration
* tskip: int, number of timesteps to skip before saving
* f: optional, string, filename to save results
* ts: boolean, True: return the timeseries, False: return the endpoint of integration
* seed: int, seed for random number generator
'''

def run_timeseries_noise(params, noise=0.05, model = MODEL.GLV, noise_implementation=NOISE.LANGEVIN_LINEAR,
                         dt=0.01, T=100, tskip = 0,
                         f=0, ts = True, seed=None):

    # Set seed for random number generator
    if seed == None:
        np.random.seed(int(time.time()))
    else:
        np.random.seed(seed)

    # Set parameters.
    if model == MODEL.GLV:
        # Verify if all parameters are given, otherwise raise error.
        try:
            omega, mu, g, initcond = params['interaction_matrix'], params['immigration_rate'], \
                                     params['growthrate'], params['initial_condition']
        except:
            for par in ['interaction_matrix', 'immigration_rate', 'growthrate', 'initial_condition']:
                if not par in params:
                    raise KeyError('Parameter %s needs to be specified for the %s model and %s noise implementation.'
                                   % (par, model.name, noise.name))

        Nspecies = len(omega) # number of species
        Nmetabolites = 0 # number of metabolites, 0 in the GLV models
        x = np.copy(initcond) # set initial state
    elif model == MODEL.QSMI:
        # Verify if all parameters are given, otherwise raise error.
        try:
            psi, d, g, dm, kappa, phi, initcond = (params['psi'], params['d'], params['g'], params['dm'],
                                                   params['kappa'], params['phi'], params['initcond'])
        except:
            for par in ['psi', 'd', 'g', 'dm', 'kappa', 'phi', 'initcond']:
                if not par in params:
                    raise KeyError('Parameter %s needs to be specified for the %s model and %s noise implementation.' % (
                    par, model.name, noise.name))

        Nspecies = len(d) # number of species
        Nmetabolites = len(dm) # number of metabolites
        x = np.copy(initcond)[:len(d)] # initial state species
        y = np.copy(initcond)[len(d):] # initial state metabolites

    # Write down header in file
    if f != 0:
        with open(f, "a") as file:
            file.write("time")
            for k in range(1, Nspecies + 1):
                file.write(",species_%d" % k)
            for k in range(1, Nmetabolites + 1):
                file.write(",metabolite_%d" % k)

            file.write("\n")

            file.write("%.3E" % 0)
            for k in initcond:
                file.write(",%.3E" % k)
            file.write("\n")

    # To save all points in timeseries, make new variable x_ts
    if ts == True:
        x_ts = np.copy(x)

    # If noise is Ito, first generate brownian motion.
    if noise_implementation == NOISE.ARATO_LINEAR:
        xt = np.zeros_like(initcond)
        bm = brownian(np.zeros(len(initcond)), int(T /dt), dt, 1, out=None)

    # Integrate ODEs according to model and noise
    for i in range(1, int(T / dt)):
        if model == MODEL.GLV:
            if noise_implementation == NOISE.LANGEVIN_LINEAR:
                dx = (omega.dot(x) * x + mu + g * x) * dt
                x += dx + noise * x * np.sqrt(dt) * np.random.normal(0, 1, x.shape)
            elif noise_implementation == NOISE.LANGEVIN_SQRT:
                x += (omega.dot(x) * x + mu + g * x) * dt + noise * np.sqrt(x) * np.sqrt(dt) * np.random.normal(0, 1, x.shape)
            elif noise_implementation == NOISE.SQRT_MILSTEIN:
                dW = np.sqrt(dt)*np.random.normal(0,1, x.shape)
                x += (omega.dot(x) * x + mu + g * x) * dt + np.sqrt(noise * x) * dW + noise**2 / 4 * (dW**2 - dt**2)
            elif noise_implementation == NOISE.LANGEVIN_CONSTANT:
                x += (omega.dot(x) * x + mu + g * x) * dt + noise * np.sqrt(dt) * np.random.normal(0, 1, x.shape)
            elif noise_implementation == NOISE.RICKER_LINEAR:
                if noise == 0:
                    b = np.ones(x.shape)
                else:
                    b = np.exp(noise * np.sqrt(dt) * np.random.normal(0, 1, x.shape))
                x = b * x * np.exp(omega.dot(x + np.linalg.inv(omega).dot(g)) * dt)
            elif noise_implementation == NOISE.ARATO_LINEAR:
                xt += x * dt

                t = i * dt

                Y = g * t - noise ** 2 / 2 * t + omega.dot(xt) + noise * bm[:, i].reshape(
                    x.shape)  # noise * np.random.normal(0, 1, initcond.shape)
                x = initcond * np.exp(Y)

            x = x.clip(min=0)

        if model == MODEL.QSMI:
            if noise_implementation == NOISE.LANGEVIN_CONSTANT:
                dx = x*(psi.dot(y) - d)
                dy = g - dm*y - y*kappa.dot(x) + ((phi.dot(x)).reshape([Nmetabolites,Nmetabolites])).dot(y)

                x += dx*dt
                y += dy*dt #+ noise * np.sqrt(dt) * np.random.normal(0, 1, y.shape)

                x = x.clip(min=0)
                y = y.clip(min=0)

        # Save abundances
        if f != 0 and i % (tskip + 1) == 0:
            with open(f, "a") as file:
                file.write("%.5E" % (i * dt))
                for k in x:
                    file.write(",%.5E" % k)
                if model == MODEL.QSMI:
                    for k in y:
                        file.write(",%.5E" % k)
                file.write("\n")
        if ts == True and i % (tskip + 1) == 0:
            x_ts = np.hstack((x_ts, x))

    # return timeseries if ts = True, else return only endpoint
    if ts == True:
        x_ts = np.vstack((dt * (tskip+1)*np.arange(len(x_ts[0]))[np.newaxis, :], x_ts))
        x_ts = pd.DataFrame(x_ts.T, columns=['time'] + ['species_%d' % i for i in range(1,Nspecies+1)])
        return x_ts
    else:
        return x


def main(seed=None,noise=0.05,sigma=0.05):
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
    ts = run_timeseries_noise(params,seed=seed,noise=noise)

    # To save the timeseries in file and read the timeseries.
    if False:
        run_timeseries_noise(params, f='my_timeseries.csv')
        ts = pd.read_csv(ts)

    # Plot the timeseries
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(1, Nspecies+1):
        ax.plot(ts['time'], ts['species_%d' % i], label='species %d' % i)

    ax.legend()
    plt.show()

    params['timeseries'] = ts
    return params

if __name__ == "__main__":
    main()