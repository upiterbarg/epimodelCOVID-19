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

def get_SIR(x, y, y0, country, forecast_len=0):
    '''
    If 'forecast_len' is nonzero, attempts to load a trace corresponding to the
    country of interest from the directory 'traces' and retrieves predicted numbers
    of infected and susceptible patients 'forecast_len' days into the future after the 
    1st case is detected in the country.
    '''

    # If in 'prediction mode', modify x, y to reflect forecast length
    if forecast_len != 0:
        ext = np.arange(1, forecast_len+1).astype(float)
        ext += x[-1]
        x = np.append(x, ext)
        y = np.empty((x.shape[0], y.shape[1]))

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

    load_dir = osp.join('traces', country.lower())

    with pm.Model() as model:
        sigma = pm.HalfNormal('sigma', 3, shape=2)

        # R0 is bounded below by 1 because we see an epidemic has occured
        R0 = pm.Bound(pm.Normal, lower=1)('R0', 2, 3)

        lmbda = pm.Normal('lambda', 0.11, 0.1)

        beta = pm.Deterministic('beta', lmbda * R0)

        print('Setting up model for '+country)
        sir_curves = sir_ode(y0=y0, theta=[beta, lmbda])

        y_obs = pm.Normal('y_obs', mu=sir_curves, sigma=sigma, observed=y)

        if forecast_len == 0: 
            trace = pm.sample(3000, tune=1500, cores=2, chains=2, target_accept=0.9, progressbar=True)

            # Save trace
            pm.save_trace(trace, load_dir, overwrite=True)

            # Get the posterior
            post= pm.sample_posterior_predictive(trace, progressbar=True)
        else:
            # Load trace
            print('Loading trace')
            trace = pm.load_trace(load_dir)

            print('Computing posterior')
            # Get posterior
            post = pm.sample_posterior_predictive(trace[500:], progressbar=True)

   
    return trace, post['y_obs'], x

def perform_inference(country, forecast_len=0):
    # If in training mode, examine data only after 100 cases have 
    # been recorded in country (for better chances at convergence).
    # Otherwise, start at the first case
    '''
    if forecast_len != 0:
        case_thresh = 100 
    else:
        case_thresh = 1
    '''
    #case_thresh = 100
    case_thresh = 10
    dd, labels = unpack_data()
    dates, x, infected, susceptible = clean_by_country(country, dd, labels, case_thresh=case_thresh)
    
    # For now, we use all but the last 'testdim' days as training data, and 
    # the remaining days as testing data.
    testdim = 1
    
    x_train = x[:-testdim]
    x_test = x[-testdim:]

    #x_train_scale = MinMaxScaler().fit_transform(x_train.reshape(-1,1)).flatten()
    
    y_train = np.hstack((susceptible[:-testdim].reshape(-1,1), infected[:-testdim].reshape(-1,1)))
    y_test = np.hstack((susceptible[-testdim:].reshape(-1,1), infected[-testdim:].reshape(-1,1)))

    y0 = [y_train[0][0], y_train[0][1]]


    trace, post, x_out = get_SIR(x_train, y_train, y0, country, forecast_len=forecast_len)

    pdb.set_trace()
    with open(country+'_post.pkl', 'wb') as buff:              
        pickle.dump({'post': post}, buff)
    
    return post, y_train, x_out, dates

def main():
    country = 'US'
    post, y_train, x_out, dates = perform_inference(country, forecast_len=0)
    #plot_post(post, dates, y_train)
    plot_SIR_curve(post, y_train, x_out, dates, country)


if __name__ == '__main__':
    main()