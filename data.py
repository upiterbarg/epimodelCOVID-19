import numpy as np 
import pandas as pd
import os
import os.path as osp
import pdb
import csv


def unpack_data():
    '''
    Accesses the submodule 'COVID-19' and processes data

    *** cloned from the repo maintained by JHU CSSE, available here: 
    https://github.com/CSSEGISandData/COVID-19
    '''

    # Unpack latest timeseries data
    ts_path = osp.join(os.getcwd(), 'COVID-19', 'csse_covid_19_data', 'csse_covid_19_time_series')
    conf_path = osp.join(ts_path, 'time_series_19-covid-Confirmed.csv')
    deaths_path = osp.join(ts_path, 'time_series_19-covid-Deaths.csv')
    recov_path = osp.join(ts_path, 'time_series_19-covid-Recovered.csv')
    confglob_path = osp.join(ts_path, 'time_series_covid19_confirmed_global.csv')
    deathsglob_path = osp.join(ts_path, 'time_series_covid19_deaths_global.csv')

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

    # Load global confirmed cases data 
    confglob, cgloc, labels_global = [], [], None
    with open(confglob_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if labels_global is None: labels_global = row
            else:
                cgloc.append(row[:4])
                confglob.append([float(r) for r in row[4:]])
    labels_global, confglob, cgloc = np.array(labels_global), np.array(confglob), np.array(cgloc)

    # Load global deaths cases data 
    deathsglob, dgloc, f = [], [], False
    with open(deathsglob_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if not f: f=True
            else:
                dgloc.append(row[:4])
                deathsglob.append([float(r) for r in row[4:]])
    deathsglob, dgloc = np.array(deathsglob), np.array(dgloc)

    # Drop diamond 'princess, canada' and 'recovered, canada' from global data 
    mask1 = cgloc[:, :2] == ['Diamond Princess','Canada']
    mask2 = cgloc[:, :2] == ['Recovered','Canada']
    mask = np.array(mask1+mask2, dtype=int).sum(axis=1) != 2
    confglob, cgloc = confglob[mask], cgloc[mask]
    deathsglob, dgloc = deathsglob[mask], dgloc[mask]

    dd = {'deaths': deaths, 'conf': conf, 'recov': recov, 'confglob': confglob, 'deathsglob': deathsglob,
        'dlocs': dlocs, 'clocs': clocs, 'rlocs': rlocs, 'cgloc': cgloc, 'dgloc': dgloc}
    return dd, labels, labels_global

def clean_by_country(country, dd, labels, case_thresh=1):
    '''
    Extract country-specific according to the specified case threshold ('case_thresh'),
    returning all relevant values (and corresponding dates) normalized by country population.

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

    # Convert data (deaths, conf, recov) from corresponding to per day reports to instead
    # represent cumulative numbers
    for i in range(1, dates.shape[0]):
        deaths[i] += deaths[i-1]
        conf[i] += conf[i-1]
        recov[i] += recov[i-1]

    # Extract total population from World Bank Data csv (note: based on 2018!!)
    df_pop = pd.read_csv("population_worldbank2018.csv")
    pop = df_pop['Population'][df_pop['Country'] == country]
    if pop.shape[0] == 0:
        raise ValueError('Failed to find country population in World Bank dataset.')
    pop = pop.values[0]
    
    # Compute fraction of infected and susceptible individuals in country by day
    infected = (conf - deaths - recov)/pop
    susceptible = (pop - conf)/pop

    return dates, np.arange(1, deaths.shape[0]+1).astype(float), infected, susceptible

def clean_by_state(state, dd, labels, case_thresh=10):
    pass



