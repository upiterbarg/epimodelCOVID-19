import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import arviz as az 
import pymc3 as pm 
import pdb
from scipy.integrate import odeint


rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def plot_params(trace, post, country):
    fig = plt.figure()
    pm.plot_posterior(trace[:500], credible_interval=0.95)
    plt.show()

def plot_post(a, dates, y_train, country):
    '''
    Use as a check of model fit.
    '''
    y_train = y_train[:-1]
    y0 = a[:,:y_train.shape[0],0]
    y1 = a[:,:y_train.shape[0],1]

    y0_mean = np.mean(y0, axis=0)
    y1_mean = np.mean(y1, axis=0)

    y0_std = np.std(y0, axis=0)*2
    y1_std = np.std(y1, axis=0)*2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

    dates = np.arange(0, y0_mean.shape[0])
    ax1.fill_between(dates, y0_mean-y0_std, y0_mean+y0_std, alpha=0.5, color='g')
    ax1.plot(dates, y0_mean, ':g', label='predicted susceptible')
    ax1.plot(dates, y_train[:, 0], 'g', label='true susceptible')

    ax1.set_xlabel(r'Days since hundredth case is reported')
    ax1.set_ylabel(r'Proportion of total population')
    ax1.legend(loc='lower left')

    ax2.fill_between(dates, y1_mean-y1_std, y1_mean+y1_std, alpha=0.5, color='b')
    ax2.plot(dates, y1_mean, ':b', label='predicted infected')
    ax2.plot(dates, y_train[:, 1], 'b', label='true infected')

    ax2.set_xlabel('Days since hundredth case is reported')
    ax2.set_ylabel(r'Proportion of total population')
    ax2.legend(loc='upper left')
    
    fig.tight_layout(pad=3.0)

    fig.suptitle(r'Inferred SIR Curves for COVID-19 Epidemic in '+country + ' vs. Training Data')
    fig.savefig('inference_inf_'+country+'_p2-8-20', dpi=1000, bbox_inches='tight')

def plot_SIR_curve(post, y_train, x_out, dates, country):
    sus, inf = post[:,:,0], post[:,:,1] # susceptible, infected
    recov = 1 - post[:,:,0] - post[:,:,1] # recovered

    # Extract training data
    sus_train = y_train[:, 0]
    inf_train = y_train[:, 1]
    recov_train = 1 - sus_train - inf_train
    
    # Compute means
    sus_mean = np.mean(sus, axis=0)
    inf_mean = np.mean(inf, axis=0)
    recov_mean = np.mean(recov, axis=0)
    # Compute standard deviations
    sus_std = np.std(sus, axis=0)*2
    inf_std = np.std(inf, axis=0)*2
    recov_std = np.std(recov, axis=0)*2


    # Enumerate prediction dates
    pred_enum = np.arange(1, x_out.shape[0]+1)
    dates_enum = np.arange(1, sus_train.shape[0]+1)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot susceptible
    plt.fill_between(pred_enum, sus_mean+sus_std, sus_mean-sus_std, alpha=0.5, color='g')
    plt.plot(dates_enum, sus_train, 'g', alpha=0.6, label=r'susceptible')
    plt.plot(pred_enum, sus_mean, ':g', alpha=0.6)
    plt.scatter(dates_enum[-1], sus_train[-1], color='g', alpha=0.6)

    # Plot infected
    plt.fill_between(pred_enum, inf_mean+inf_std, inf_mean-inf_std, alpha=0.5, color='b')
    plt.plot(dates_enum, inf_train, 'b', alpha=0.6, label=r'infected')
    plt.plot(pred_enum, inf_mean, ':b', alpha=0.6)
    plt.scatter(dates_enum[-1], inf_train[-1], color='b', alpha=0.6)

    # Plot recovered
    plt.fill_between(pred_enum, recov_mean+recov_std, recov_mean-recov_std, alpha=0.5, color='r')
    plt.plot(dates_enum, recov_train, 'r', alpha=0.6, label=r'removed')
    plt.plot(pred_enum, recov_mean, ':r', alpha=0.6)
    plt.scatter(dates_enum[-1], recov_train[-1], color='r', alpha=0.6)
    
    plt.xlabel(r'Days since hundredth case is reported')
    plt.ylabel(r'Proportion of total population')
    plt.ylim([0, 1.01])
    plt.xlim([pred_enum[0], pred_enum[-1]])
    plt.title(r'Forecasting the Next 300 Days of the COVID-19 Epidemic in '+country+' with SIR')
    ax.legend(loc='upper right')

    fig.savefig(country+'_sir_plot', dpi=1000, bbox_inches='tight')

    return