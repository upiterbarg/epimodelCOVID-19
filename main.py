import pymc3 as pm 
from pymc3.ode import DifferentialEquation
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import arviz as az 
import pdb
from data import *

plt.style.use('seaborn-darkgrid')

# SIR Model
def SIR(y, t, p):
    ds = -p[0]*y[0]*y[1]
    di = p[0]*y[0]*y[1] - p[1]*y[1]
    return [ds, di]

sir_model = DifferentialEquation(
    func=SIR,
    times=np.arange(0.25, 5, 0.25),
    n_states=2,
    n_theta=2,
    t0=0,
)

def main():
    dd, labels = unpack_data()
    date_labels = labels[4:]

    deaths, conf, recov = dd['deaths'].sum(axis=1), dd['conf'].sum(axis=1), dd['recov'].sum(axis=1)

    times = np.arange(0,len(date_labels),1)
    pdb.set_trace()
    beta,gamma = 4,1.0
    

    with pm.Model() as model:
        sigma = pm.HalfCauchy('sigma', 1, shape=2)

        # R0 is bounded below by 1 because we see an epidemic has occured
        R0 = pm.Bound(pm.Normal, lower=1)('R0', 2,3)
        lam = pm.Lognormal('lambda',pm.math.log(2),2)
        beta = pm.Deterministic('beta', lam*R0)

        sir_curves = sir_model(y0=[0.99, 0.01], theta=[beta, lam])

        Y = pm.Lognormal('Y', mu=pm.math.log(sir_curves), sd=sigma, observed=)

        prior = pm.sample_prior_predictive()
        trace = pm.sample(1000,tune=500, target_accept=0.9, cores=1)
        posterior_predictive = pm.sample_posterior_predictive(trace)

        data = az.from_pymc3(trace=trace, prior = prior, posterior_predictive = posterior_predictive)

        fig, ax = plt.subplots()
        az.plot_posterior(data,round_to=2, credible_interval=0.95)
        fig.savefig('confirmed', dpi=1000, bbox_inches='tight')


if __name__ == '__main__':
    main()