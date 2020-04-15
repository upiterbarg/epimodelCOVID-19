# epimodelCOVID-19

This repository uses PyMC3 to estimate the posterior distributions corresponding to the parameters 'R_0' and 'lambda' for the SIR (Susceptible-Infective-Removed) compartmental epidemic model applied to the COVID-19 epidemic in certain countries (with plots provided reflexting the epidemic in Italy and China, trained on data collected prior to 03/27/20).

Data, which can be found in the folder 'COVID-19' is synced with the repository '2019 Novel Coronavirus COVID-19 (2019-nCoV) Data Repository by Johns Hopkins CSSE': https://github.com/CSSEGISandData/COVID-19.

This project was inspired by the GSoC 2019 PyMC3 tutorial on estimation of ODE parameter posteriors here:  https://docs.pymc.io/notebooks/ODE_API_introduction.html

A YAML file listing all dependencies is provided.
