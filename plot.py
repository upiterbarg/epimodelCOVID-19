import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rc
import arviz as az 
import pdb

#plt.style.use('seaborn-darkgrid')
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

    dates = np.arange(1, y0_mean.shape[0]+1)
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

def plot_SIR_curve(post, y_train, x_out, dates, country, SIR=False):
    if SIR:
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

        #pdb.set_trace()

        # Enumerate prediction dates
        pdb.set_trace()
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
        plt.plot(dates_enum, recov_train, 'r', alpha=0.6, label=r'recovered')
        plt.plot(pred_enum, recov_mean, ':r', alpha=0.6)
        plt.scatter(dates_enum[-1], recov_train[-1], color='r', alpha=0.6)
        
        plt.xlabel(r'Days since hundredth case is reported')
        plt.ylabel(r'Proportion of total population')
        plt.ylim([0, 1.01])
        plt.xlim([pred_enum[0], pred_enum[-1]])
        plt.title(r'Inferred SIR Model for COVID-19 Epidemic in '+country)
        ax.legend(bbox_to_anchor=(1.2, 1.1))

        fig.savefig(country+'_sir_plot', dpi=1000, bbox_inches='tight')
    else:
        sus, inf, exp = post[:,:,0], post[:,:,1], post[:, :, 2] # susceptible, infected
        recov = 1 - post[:,:,0] - post[:,:,1] - post[:, :, 2] # recovered

        total = sus + inf + exp # infected + recovered + exposed = all recorded cases
        grad = np.gradient(total, axis=1) # apply gradient on total cases

        # Extract training data
        sus_train = y_train[:, 0]
        inf_train = y_train[:, 1]
        exp_train = y_train[:, 2]
        recov_train = 1 - sus_train - inf_train - exp_train
        
        # Compute means
        sus_mean = np.mean(sus, axis=0)
        inf_mean = np.mean(inf, axis=0)
        recov_mean = np.mean(recov, axis=0)
        exp_mean = np.mean(exp, axis=0)
        total_mean = np.mean(total, axis=0)
        grad_mean = np.mean(grad, axis=0)

        # Compute standard deviations
        sus_std = np.std(sus, axis=0)
        inf_std = np.std(inf, axis=0)
        recov_std = np.std(recov, axis=0)
        exp_std = np.std(exp, axis=0)
        total_std = np.std(total, axis=0)
        grad_std = np.std(grad, axis=0)

        #pdb.set_trace()

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
        plt.plot(dates_enum, recov_train, 'r', alpha=0.6, label=r'recovered')
        plt.plot(pred_enum, recov_mean, ':r', alpha=0.6)
        plt.scatter(dates_enum[-1], recov_train[-1], color='r', alpha=0.6)

        # Plot exposed
        plt.fill_between(pred_enum, exp_mean+exp_std, exp_mean-exp_std, alpha=0.5, color='m')
        plt.plot(dates_enum, exp_train, 'm', alpha=0.6, label=r'exposed')
        plt.plot(pred_enum, exp_mean, ':m', alpha=0.6)
        plt.scatter(dates_enum[-1], exp_train[-1], color='m', alpha=0.6)
        
        plt.xlabel(r'Days since hundredth case is reported')
        plt.ylabel(r'Proportion of total population')
        plt.ylim([0, 1.01])
        plt.xlim([pred_enum[0], pred_enum[-1]])
        plt.title(r'Inferred SEIR Model for COVID-19 Epidemic in '+country)
        ax.legend(bbox_to_anchor=(1.2, 1.1))

        fig.savefig(country+'_seir_plot', dpi=1000, bbox_inches='tight')


    return