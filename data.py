import numpy as np 
import h5py
import os
import os.path as osp
import pdb
import csv


def unpack_data():
    # Unpack latest timeseries data
    ts_path = osp.join(os.getcwd(), 'COVID-19', 'csse_covid_19_data', 'csse_covid_19_time_series')
    deaths_path = osp.join(ts_path, 'time_series_19-covid-Confirmed.csv')
    conf_path = osp.join(ts_path, 'time_series_19-covid-Deaths.csv')
    recov_path = osp.join(ts_path, 'time_series_19-covid-Recovered.csv')

    # Load deaths data
    deaths, labels, dlocs = [], [], []
    with open(deaths_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if len(labels) == 0: labels=row
            else:
                dlocs.append(row[:4])
                deaths.append([float(r) for r in row[4:]])
    deaths, dlocs = np.array(deaths), np.array(dlocs)

    # Load confirmed cases data
    conf, clocs, f = [], [], False
    with open(conf_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if not f: f=True
            else:
                clocs.append(row[:4])
                conf.append([float(r) for r in row[4:]])
    conf, clocs = np.array(conf), np.array(clocs)

    # Load recovered data
    recov, rlocs, f = [], [], False
    with open(recov_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if not f: f=True
            else:
                rlocs.append(row[:4])
                recov.append([float(r) for r in row[4:]])
    recov, rlocs = np.array(recov), np.array(rlocs)

    dd = {'deaths': deaths, 'conf': conf, 'recov': recov, 'dlocs':dlocs, 'clocs':clocs, 'rlocs':rlocs}
    return dd, labels

