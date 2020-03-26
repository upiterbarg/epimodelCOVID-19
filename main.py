import pymc3 as pm 
from pymc3.ode import DifferentialEquation
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.integrate import odeint, solve_ivp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import arviz as az 
import pdb
import theano
import pickle
import os.path as osp

from data import *
from plot import *

floatX = theano.config.floatX

def get_SIR(x, y, y0, country, forecast_len=0, load=False):
    # SIR Model
    def SIR(y, t, p):
        ds = -p[0]*y[0]*y[1] # Susceptible differential
        di = p[0]*y[0]*y[1] - p[1]*y[1] # Infected differential
        return [ds, di]

    # Initialize ODE
    sir_ode = DifferentialEquation(
        func=SIR,
        times=x,
        n_states=2, 
        n_theta=2, 
        t0=0
    )

    load_dir = osp.join('traces', country.lower().replace(' ','_'))

    with pm.Model() as model:
        sigma = pm.HalfNormal('sigma', 3, shape=2)

        # R0 is bounded below by 1 because we see an epidemic has occured
        R0 = pm.Bound(pm.Normal, lower=1)('R0', 2, 3)

        lmbda = pm.Normal('lambda', 0.11, 0.1)

        beta = pm.Deterministic('beta', lmbda * R0)

        print('Setting up model')
        sir_curves = sir_ode(y0=y0, theta=[beta, lmbda])

        y_obs = pm.Normal('y_obs', mu=sir_curves, sigma=sigma, observed=y)

        if not load: 
            trace = pm.sample(3000, tune=1500, cores=12, chains=12, target_accept=0.9, progressbar=True)

            # Save trace
            pm.save_trace(trace, load_dir, overwrite=True)
        else:
            # Load trace
            trace = pm.load_trace(load_dir)

        pdb.set_trace()
        posterior_predictive = pm.sample_posterior_predictive(trace, progressbar=True)

    return trace, posterior_predictive

def perform_inference(country, case_thresh, forecast_len=0, load=False):
    dd, labels, labels_global = unpack_data()
    dates, x, infected, susceptible = clean_by_country(country, dd, labels, case_thresh=case_thresh)
    
    # For now, we use all but the last 'testdim' days as training data, and 
    # the remaining days as testing data.
    testdim = 1
    
    x_train = x[:-testdim]
    x_test = x[-testdim:]

    x_train_scale = MinMaxScaler().fit_transform(x_train.reshape(-1,1)).flatten()

    y_train = np.hstack((susceptible[:-testdim].reshape(-1,1), infected[:-testdim].reshape(-1,1)))
    y_test = np.hstack((susceptible[-testdim:].reshape(-1,1), infected[-testdim:].reshape(-1,1)))

    y0 = [y_train[0][0], y_train[0][1]]

    trace, posterior_predictive = get_SIR(x_train_scale, y_train, y0, country, forecast_len=forecast_len, load=load)

    print(pm.summary(trace))

    return posterior_predictive, x_train, y_train

def main():
    posterior_predictive, dates, y_train = perform_inference('China', 100, load=False)
    plot_post(posterior_predictive, dates, y_train)


if __name__ == '__main__':
    main()