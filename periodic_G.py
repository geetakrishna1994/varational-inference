import numpy as np
import emcee
import corner
import pymc3 as pm
from scipy import stats
from scipy import optimize
from scipy import integrate
from matplotlib import pyplot as plt

# Dataset for this project (Obtained from Pitkin (https://github.com/mattpitkin/periodicG))
Ggrad = (6.676e-11-6.672e-11)/(372.31-96.28)
UWup = 6.672e-11 + Ggrad*(249.17-96.28)
UWuperr = (316.94-249.17)*Ggrad

UWup = 6.67421e-11
UWuperr = 0.00098e-11

G = np.array([6.67248e-11, 6.67398e-11, 6.67228e-11, 6.674255e-11, 6.67559e-11, UWup, 6.67387e-11, 6.67234e-11, 6.674252e-11, 6.67554e-11, 6.67349e-11, 6.67191e-11])
Err = np.array([0.00043e-11, 0.00070e-11, 0.00087e-11, 0.000092e-11, 0.00027e-11, UWuperr, 0.00027e-11, 0.00014e-11, 0.000120e-11, 0.00016e-11, 0.00018e-11, 0.00099e-11])
Year = np.array([1981.90,1996.97,1998.32,2000.46,2001.16,2002.02,2003.39,2004.40,2006.48,2007.68,2009.17,2013.57])
relYear = Year - Year[0]

posgrad = (2015.-1981.) / (674.04-92.79)
positions = np.array([108.17, 365.87, 388.94, 425.48, 437.50, 452.08, 475.48, 492.79, 528.44, 552.61, 574.52, 649.52])
positions = positions-92.79
years = positions*posgrad
t_0 = (420.83-92.79)*posgrad

# Range of parameters
meanG = np.mean(G)
sigma_meanG = np.std(G) / np.sqrt(len(G))  
muGmin = meanG - 6.*sigma_meanG
muGmax = meanG + 6.*sigma_meanG

sigmasysmax = np.max(G)-np.min(G)
sigmasysmin = np.min(Err)

Amin = sigmasysmin
Amax = sigmasysmax

timediff = np.diff(np.sort(years)[1:]) # skip the first longer time

periodmin = 5.90
periodmax = 5.90

phimin = 0
phimax = 0

priorSet = [muGmin,muGmax,sigmasysmin,sigmasysmax,Amin,Amax,periodmin,periodmax,phimin,phimax]
data = [G,Err,years]

with pm.Model() as hyp_1:
    muG = pm.Uniform('muG', lower=muGmin, upper=muGmax)
    y,sigma_y,time = data
    mu = np.repeat(muG, len(y))
    y_obs = pm.Normal('y_obs',mu=mu,sd=sigma_y,observed=y)

with hyp_1:
    approx1 = pm.fit(n=100000,method='advi')
    trace1 = pm.sample_approx(approx1)

Z1 = np.exp(np.array([hyp_1.logp(pt) for pt in trace1])).mean()
print(np.log(Z1))

with pm.Model() as hyp_2:
    muG = pm.Uniform('muG', lower=muGmin, upper=muGmax)
    sigmasys = pm.Uniform('sigmasys',lower=sigmasysmin,upper=sigmasysmax)
    mu = np.repeat(muG, len(y))
    y,sigma_y,time = data
    sd = np.sqrt(sigma_y**2 + sigmasys**2)
    y_obs = pm.Normal('y_obs',mu=mu,sd=sd,observed=y)
with hyp_2:
    approx2 = pm.fit(n=100000,method='advi')
    trace2 = pm.sample_approx(approx2)

Z2 = np.exp(np.array([hyp_2.logp(pt) for pt in trace2])).mean()
print(np.log(Z2))

with pm.Model() as hyp_3:
    y,sigma_y,time = data
    muG = pm.Uniform('muG', lower=muGmin, upper=muGmax)
    A = pm.Uniform('A', lower=Amin,upper=Amax)
    P = 5.90
    phi = 0
    mu = muG + A * np.sin(phi + 2 * np.pi * time / P)
    y_obs = pm.Normal('y_obs',mu=mu,sd=sigma_y,observed=y)
with hyp_3:
    approx3 = pm.fit(n=100000,method='advi')
    trace3 = pm.sample_approx(approx3)

Z3 = np.exp(np.array([hyp_3.logp(pt) for pt in trace3])).mean()
print(np.log(Z3))

with pm.Model() as hyp_4:
    y,sigma_y,time = data
    muG = pm.Uniform('muG', lower=muGmin, upper=muGmax)
    sigmasys = pm.Uniform('sigmasys',lower=sigmasysmin,upper=sigmasysmax)
    A = pm.Uniform('A', lower=Amin,upper=Amax)
    P = 5.90
    phi = 0
    mu = muG + A * np.sin(phi + 2 * np.pi * time / P)
    sd = np.sqrt(sigma_y**2 + sigmasys**2)
    y_obs = pm.Normal('y_obs',mu=mu,sd=sd,observed=y)
with hyp_4:
    approx4 = pm.fit(n=100000,method='advi')
    trace4 = pm.sample_approx(approx4)

Z4 = np.exp(np.array([hyp_4.logp(pt) for pt in trace4])).mean()
print(np.log(Z4))

print('Z1:',np.log(Z1))
print('Z2:',np.log(Z2))
print('Z3:',np.log(Z3))
print('Z4:',np.log(Z4))