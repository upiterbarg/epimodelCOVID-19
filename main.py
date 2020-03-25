import pymc3 as pm 
from pymc3.ode import DifferentialEquation
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.integrate import odeint, solve_ivp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import arviz as az 
import pdb
from data import *
import theano

floatX = theano.config.floatX
seed(0)
#plt.style.use('seaborn-darkgrid')
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

# SIR Model
def SIR(y, t, p):
    ds = -p[0]*y[0]*y[1] # Susceptible differential
    di = p[0]*y[0]*y[1] - p[1]*y[1] # Infected differential
    return [ds, di]

class DE(DifferentialEquation):
    def _simulate(self, y0, theta):
        # Begin with initial conditions and raveled sensitivity matrix
        s0 = np.concatenate([y0, self._sens_ic])
        
        # Integrate
        sol = solve_ivp(
            fun = lambda t, Y: self._system(Y, t, tuple(np.concatenate([y0, theta]))),
            t_span=[self._augmented_times.min(), self._augmented_times.max()],
            y0=s0,
            method='RK23',
            t_eval=self._augmented_times[1:],
            atol=1, rtol=1,
            max_step=0.02).y.T.astype(floatX)
        
        # Extract the solution
        y = sol[:, :self.n_states]

        # Reshape sensativities
        sens = sol[0:, self.n_states:].reshape(self.n_times, self.n_states, self.n_p)

        return y, sens


def perform_inference(country):
    dd, labels, labels_global = unpack_data()
    dates, x, infected, susceptible = clean_by_country(country, dd, labels, case_thresh=10)

    # For now, we use all but the last 'testdim' days as training data, and 
    # the remaining days as testing data.
    testdim = 2
    
    train = x[:-testdim]
    test = x[-testdim:]

    x_scale =  MinMaxScaler().fit_transform(train.reshape(-1,1)).flatten()

    y_train = np.hstack((susceptible[:-testdim].reshape(-1,1), infected[:-testdim].reshape(-1,1)))
    y_test = np.hstack((susceptible[-testdim:].reshape(-1,1), infected[-testdim:].reshape(-1,1)))

    y0 = [y_train[0][0], y_train[0][1]]

    sir_model = DE(
        func=SIR,
        times=train,
        n_states=2,
        n_theta=2, 
        t0=0
    )

    with pm.Model() as model:
        print('Initializing Priors')
        sigma = pm.HalfNormal('sigma', 1, shape=2)

        # R0 is bounded below by 1 because we see an epidemic has occured
        R0 = pm.Bound(pm.Normal, lower=1)('R0', 2, 3)
        lmbda = pm.Normal('lambda', 0, 10)
        delta = pm.Normal('delta', 0, 10)
        beta = pm.Deterministic('beta', lmbda*R0)
        
        
        print('Setting up model')
        sir_curves = sir_model(y0=y0, theta=[beta, lmbda])

        Y = pm.Normal('Y', mu=sir_curves, sigma=sigma, observed=y_train)

        print('Starting sampling')

        trace = pm.sample(2000, tune=1000, cores=12, progressbar=True)
        posterior_predictive = pm.sample_posterior_predictive(trace, progressbar=True)

    pdb.set_trace()

    print(pm.summary(trace))

    return posterior_predictive, x[:-testdim], y_train

def plot(posterior_predictive, dates, y_train):
    a = posterior_predictive['Y']
    y0 = a[:,:,0]
    y1 = a[:,:,1]

    y0_mean = np.mean(y0, axis=0)
    y1_mean = np.mean(y1, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(dates, y0_mean, ':g', label='predicted susceptible')
    plt.plot(dates, y_train[:, 0], 'g', label='true susceptible')

    ax.set_xlabel('num days since first case')
    ax.set_ylabel('fraction of population')
    ax.legend()
    fig.savefig('inference_sus', dpi=1000, bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(dates, y1_mean, ':b', label='predicted infected')
    plt.plot(dates, y_train[:, 1], 'b', label='true infected')

    ax.set_xlabel('num days since first case')
    ax.set_ylabel('fraction of population')
    ax.legend()
    fig.savefig('inference_inf', dpi=1000, bbox_inches='tight')

def main():
    posterior_predictive, dates, y_train = perform_inference('US')
    plot(posterior_predictive, dates, y_train)
    pdb.set_trace()


if __name__ == '__main__':
    main()