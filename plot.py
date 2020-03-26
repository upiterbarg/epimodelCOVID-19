import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rc
import arviz as az 

plt.style.use('seaborn-darkgrid')
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def plot_post(a, dates, y_train, country):
    '''
    Use as a check of model fit.
    '''
    y0 = a[:,:,0]
    y1 = a[:,:,1]

    y0_mean = np.mean(y0, axis=0)
    y1_mean = np.mean(y1, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(dates, y0_mean, ':g', label='predicted susceptible')
    plt.plot(dates, y_train[:, 0], 'g', label='true susceptible')

    ax.set_xlabel(r'num days since hundreth recorded case')
    ax.set_ylabel(r'fraction of population')
    ax.set_title(r'Inferred SIR: '+country)
    ax.legend()
    fig.savefig('inference_sus', dpi=1000, bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(dates, y1_mean, ':b', label='predicted infected')
    plt.plot(dates, y_train[:, 1], 'b', label='true infected')

    ax.set_xlabel('rnum days since hundreth recorded case')
    ax.set_ylabel(r'fraction of population')
    ax.set_title(r'Inferred SIR: '+country)
    ax.legend()
    fig.savefig('inference_inf', dpi=1000, bbox_inches='tight')

def plot_SIR_curve(post, y_train, x_out, dates, country):
    sus, inf = post[:,:,0], post[:,:,1] # susceptible, infected
    recov = 1 - post[:,:,0] - post[:,:,1] # recovered

    total = sus + inf # infected + recovered = all recorded cases
    grad = np.gradient(total, axis=1) # apply gradient on total cases

    # Extract training data
    sus_train = y_train[:, 0]
    inf_train = y_train[:, 1]
    recov_train = 1 - sus_train - inf_train
    
    # Compute means
    sus_mean = np.mean(sus, axis=0)
    inf_mean = np.mean(inf, axis=0)
    recov_mean = np.mean(recov, axis=0)
    total_mean = np.mean(total, axis=0)
    grad_mean = np.mean(grad, axis=0)

    # Compute standard deviations
    sus_std = np.std(sus, axis=0)
    inf_std = np.std(inf, axis=0)
    recov_std = np.std(recov, axis=0)
    total_std = np.std(total, axis=0)
    grad_std = np.std(grad, axis=0)

    # Enumerate prediction dates
    pred_enum = np.arange(1, x_out.shape[0]+1)
    dates_enum = np.arange(1, sus_train.shape[0]+1)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot susceptible
    plt.fill_between(pred_enum, sus_mean+sus_std, sus_mean-sus_std, alpha=0.5, color='g')
    plt.plot(dates_enum, sus_train, 'g', label=r'susceptible')
    plt.plot(pred_enum, sus_mean, ':g', alpha=0.6)

    # Plot infected
    plt.fill_between(pred_enum, inf_mean+inf_std, inf_mean-inf_std, alpha=0.5, color='b')
    plt.plot(dates_enum, inf_train, 'b', label=r'infected')
    plt.plot(pred_enum, inf_mean, ':b', alpha=0.6)

    # Plot recovered
    plt.fill_between(pred_enum, recov_mean+recov_std, recov_mean-recov_std, alpha=0.5, color='r')
    plt.plot(dates_enum, recov_train, 'r', label=r'recovered')
    plt.plot(pred_enum, recov_mean, ':r', alpha=0.6)
    
    plt.xlabel(r'Days')
    plt.xlabel(r'Proportion of total population')
    plt.ylim([0, 1.01])
    plt.xlim([pred_enum[0], pred_enum[-1]])
    plt.title(r'SIR Infection Rates in '+country)
    plt.legend()

    fig.savefig(country+'_sir_plot', dpi=1000, bbox_inches='tight')

    return