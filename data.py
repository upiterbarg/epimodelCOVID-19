import numpy as np 
import pandas as pd
import os
import os.path as osp
import pdb
import csv
import scipy.stats as stats


def unpack_data():
    '''
    Accesses the submodule 'COVID-19' and processes data

    *** cloned from the repo maintained by JHU CSSE, available here: 
    https://github.com/CSSEGISandData/COVID-19
    '''

    # Unpack latest timeseries data
    ts_path = osp.join(os.getcwd(), 'COVID-19', 'csse_covid_19_data', 'csse_covid_19_time_series')
    conf_path = osp.join(ts_path, 'time_series_covid19_confirmed_global.csv')
    deaths_path = osp.join(ts_path, 'time_series_covid19_deaths_global.csv')
    recov_path = osp.join(ts_path, 'time_series_covid19_recovered_global.csv')

    # Load deaths data
    deaths, labels, dlocs = [], [], []
    with open(deaths_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if len(labels) == 0: labels=row[:-1]
            else:
                dlocs.append(row[:4])
                deaths.append([float(r) for r in row[4:-1]])
    labels, deaths, dlocs = np.array(labels), np.array(deaths), np.array(dlocs)

    # Load confirmed cases data
    conf, clocs, f = [], [], False
    with open(conf_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if not f: f=True
            else:
                clocs.append(row[:4])
                conf.append([float(r) for r in row[4:-1]])
    conf, clocs = np.array(conf), np.array(clocs)

    # Load recovered data
    recov, rlocs, f = [], [], False
    with open(recov_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if not f: f=True
            else:
                rlocs.append(row[:4])
                recov.append([float(r) for r in row[4:-1]])
    recov, rlocs = np.array(recov), np.array(rlocs)

    dd = {'deaths': deaths, 'conf': conf, 'recov': recov, 'dlocs': dlocs, 'clocs': clocs, 'rlocs': rlocs}
    return dd, labels

def clean_by_country(country, dd, labels, case_thresh=1, SIR=True):
    '''
    Extract country-specific according to the specified case threshold ('case_thresh'),
    returning all relevant values (and corresponding dates) normalized by country population.

    If SIR is False, estimates

    Returns:
        dates -- (type: str np.array) all relevant dates
        x -- (type: int np.array) enumeration of data for SIR model
        infected -- (type: Float32 np.array) fraction of population infected by disease by day
        susceptible -- (type: Float32 np.array) fraction of population susceptible to disease by day
         '''
    dlocs, clocs, rlocs = dd['dlocs'], dd['clocs'], dd['rlocs'] # Unpack location labels (equivalent for )
    
    # Sum across all regions (if regional info for country is given)
    deaths = dd['deaths'][dlocs[:, 1] == country, :].sum(axis=0)
    conf = dd['conf'][clocs[:, 1] == country, :].sum(axis=0)
    recov = dd['recov'][rlocs[:, 1] == country, :].sum(axis=0)

    if deaths.shape[0] == 0 or conf.shape[0] == 0 or recov.shape[0] == 0:
        raise ValueError('Failed to find country in data! Check your spelling.')

    # Apply case threshold
    mask = conf >= case_thresh
    min_ind = np.min(np.arange(0, conf.shape[0])[mask])
    deaths, conf, recov, dates = deaths[min_ind:], conf[min_ind:], recov[min_ind:], labels[4+min_ind:]


    # Estimate 'exposed' for SEIR based on publiched covid19 incubation data
    # (fairly lognormal with mean=5.5 and std=4.75)
    exposed = None
    if not SIR:
        newcases = [conf[0]]
        for i in range(1, conf.shape[0]):
            newcases.append(conf[i]-conf[i-1])
        newcases = np.array(newcases)

        total_cases = int(conf[-1])
        rincub_times = np.round(stats.lognorm.rvs(4.75, loc=5.5, size=total_cases)).astype(int)
        exposed = np.zeros(conf.shape[0])
        for i in range(total_cases):
            mask = newcases > 0
            if not True in mask: pdb.set_trace()
            min_ind = np.min(np.arange(0, newcases.shape[0])[mask])
            if min_ind - rincub_times[i] >= 0:
                exposed[min_ind - rincub_times[i]] += 1
            newcases[min_ind] -= 1

        # Apply second case threshold if SIR is True
        deaths, conf, recov, exposed, dates = deaths[:-14], conf[:-14], recov[:-14], exposed[:-14], dates[:-14]

    # Extract total population from World Bank Data csv (note: based on 2018!!)
    df_pop = pd.read_csv("population_worldbank2018.csv")
    pop = df_pop['Population'][df_pop['Country'] == country]
    if pop.shape[0] == 0:
        raise ValueError('Failed to find country population in World Bank dataset.')
    pop = pop.values[0]

    
    # Compute fraction of infected and susceptible individuals in country by day
    if exposed is None:
        infected = (conf - deaths - recov)/pop
        susceptible = (pop - conf)/pop
    else:
        infected = (conf - deaths - recov)/pop
        susceptible = (pop - conf - exposed)/pop
        exposed = exposed/pop


    return dates, np.arange(1, deaths.shape[0]+1).astype(float), infected, susceptible, exposed

def clean_by_state(state, dd, labels, case_thresh=10):
    pass



