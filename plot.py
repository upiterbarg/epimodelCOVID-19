import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rc
import arviz as az 

plt.style.use('seaborn-darkgrid')
rc('font',**{'family':'serif','serif':['Palatino']})

def plot_post(posterior_predictive, dates, y_train):
    a = posterior_predictive['y_obs']
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
    ax.set_title('Inferred SIR: Italy')
    ax.legend()
    fig.savefig('inference_sus', dpi=1000, bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(dates, y1_mean, ':b', label='predicted infected')
    plt.plot(dates, y_train[:, 1], 'b', label='true infected')

    ax.set_xlabel('num days since first case')
    ax.set_ylabel('fraction of population')
    ax.set_title('Inferred SIR: Italy')
    ax.legend()
    fig.savefig('inference_inf', dpi=1000, bbox_inches='tight')