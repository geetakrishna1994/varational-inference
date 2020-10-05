# imports
import numpy as np
np.random.seed(1234)
import scipy
import pymc3 as pm
import theano
import theano.tensor as tt
import numba
import matplotlib.pyplot as plt
import time
import warnings
import os
import pandas as pd
import seaborn as sns

sns.set_style('darkgrid')
warnings.filterwarnings('ignore')
#os.environ['PYTHONHASHSEED']=str(0)
#pm.set_tt_rng(1234)

if not os.path.exists('./plots'):
    os.mkdir('./plots')

# for solving the equation for u
@numba.vectorize("float64(float64, float64)", nopython=True)
def kepler(M, e):
    E0 = M
    E = M
    
    for i in range(50000):
        g = E0 - e * np.sin(E0) - M
        gp = 1. - e * np.cos(E0)
        E = E0 - g / gp
        if np.abs((E - E0) / E) <= 1.234e-6:
            return E
        E0 = E
    #print("returns nan")
    return np.nan

class Kepler(theano.Op):
    def make_node(self, M, e):
        M = tt.as_tensor_variable(M)
        e = tt.as_tensor_variable(e)
        return theano.Apply(self, [M, e], [M.type()])
    
    def perform(self, node, inputs, outputs):
        M, e = inputs
        e = np.float64(e)
        outputs[0][0] = kepler(M, e)
    
    def grad(self, inputs, g):
        M, e = inputs
        E = self(M, e)
        dE_dM = 1. / (1.0 - e * tt.cos(E))
        dE_de = tt.sin(E) * dE_dM
        return dE_dM * g[0], (dE_de * g).sum()
    
    def infer_shape(self, node, i0_shapes):
        return [i0_shapes[0]]

# data creation
def true_anomaly(t,tp,e,tau):
    temp1=np.min((t-tau)/tp)-1
    temp2=np.max((t-tau)/tp)+1
    u1=np.linspace(2*np.pi*temp1,2*np.pi*temp2,1000)
    ma=u1-e*np.sin(u1)
    myfunc=scipy.interpolate.interp1d(ma,u1)
    u=myfunc((2*np.pi)*(t-tau)/tp)
    return 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(0.5*u))

def vr(t,kappa,tp,e,tau,omega,v0):
    omega=np.radians(omega)
    f=true_anomaly(t,tp,e,tau)
    return kappa*(np.cos(f+omega)+e*np.cos(omega))+v0

def create_data(num_points):
    kappa=0.15
    tp=350.0
    e=0.3
    tau=87.5
    omega=-90.0
    v0=0.0
    t = np.linspace(0,tp*2,num_points)
    vr1=vr(t,kappa,tp,e,tau,omega,v0)
    err = np.random.normal(0.,0.02,num_points) % 1e-2
    return t,vr1+err,err

t,rv,rv_err = create_data(500)
plt.errorbar(x=t,y=rv,yerr=rv_err,fmt='.')
plt.savefig('./plots/data.png')
plt.clf()
print(rv.mean())
print(rv_err.mean())
# model definition
with pm.Model() as my_model:
        #balan exofit paper
    #https://watermark.silverchair.com/mnras0394-1936.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAtMwggLPBgkqhkiG9w0BBwagggLAMIICvAIBADCCArUGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMlOR-UI7QAtiAY8_CAgEQgIIChh4btrd3dwkOoOua9LWY1G4p0hj5VmbSCZTamd-_DeAexRP3y4uwHx-YAG5FPkeiUFq5voesTmdpPgLX3YQrZPK11kPRBcbe-UAL1u_Tlu-H0VdagkwU1vQ5l5LCRqL8g3Ndnqa4b-mML1xpKMvmY5vmjD9hZsM9urVahFTWcOQeYx3YFPq9qR33Zrn6Zie4uPkpmdd25BZAUlam_uTo4jp3soq73cerr2lzTHvap_o_HrPR4jmU73jaJuk9OdgQWcsSJHVpg08cC16qx7IKyn_rQsLyF1FX4JqIaRpVdHZYpTrj42CB4lYE_Df9RjMTN0h_54mXHbaFYUu-yb9I-XsGcVQVIUj8mJash6zqwc9TZ5kULaXRJ7BZa4NSUKr-4q6xp9aPID7ipRphLpFTmR_tKbPWX2wvdeu862L0JtjrSGAoR2WZZiCa1ptDmw1VFyPO0JTtzgVqsoTyxCNm7GwYrIVb0BfUBAStq9-WwjmxAUrxsvXQqlXUQ2Lfccd6yUaOn8RuXsM8rcHXk3vf8VhEyF1cs0QUxIa9WD8O6lHXh7jHD8ssvwWe4AnFlI1_JtGxqtN4X8cjp_xMXO7OfQ1YvK7bq7zb9hf4IvHCxHJRQS_4CoGvsXKLavw7ZCbXch7CVZBZG4nYnxHZnnP5tdSJZCccA3Yy05CeFaBRF2lAt3Fdmq9sPryUCJyZLt_V7DCjk53jVyop-Lp1zHb6YmO0GqMtG-2W7e4OHht9iPQR2N0oH-HRpbg4CWl-2RXyGndo6cMVKPyGzOmPn65kllxx3yICH9rIQWfqWsHTnxL3x8Zl06iZek2sYv5fs8nK5resplk8BeZcZ4USCkDtNlgDuLWtMu4
    '''
        jeffereys prior
        T = 1 / [T * ln(tmax/tmin)]
    '''
    def jeffereys_t_logp(t, tmax=15000, tmin=0.2):
        return -tt.log(t * tt.log(tmax / tmin)) 
    '''
        modified jeffereys prior
        k = 1/(k+k0) * 1/ln(1 + kmax/k0)
    '''
    def mod_jeffereys_k_logp(k, kmax=400, k0=1):
        return -tt.log((k + k0) * tt.log( 1 + kmax/k0))
    T = pm.DensityDist("T", jeffereys_t_logp,transform=pm.distributions.transforms.log, testval = 350.)
    K = pm.DensityDist("K", mod_jeffereys_k_logp,transform=pm.distributions.transforms.Log(), testval=0.15)
    e = pm.Uniform('e', lower=0., upper=1., testval=0.3)
    w = pm.Uniform("w", lower=-2 * np.pi, upper=2 * np.pi, testval=-np.pi/2)
    tau = pm.Uniform("tau", lower=0, upper=2 * np.pi, testval=np.radians(87.5))
    M = 2 * (np.pi / T) * t - tau
    u = Kepler()(M,e)
    f = 2 * tt.arctan2(tt.sqrt(1 + e) * tt.tan(0.5 * u), tt.sqrt(1 - e))
    v0 = pm.Uniform('v0', lower=-1, upper=1)
    v_mean = v0 + K * (tt.cos(f + w) + e * tt.cos(w))
    jitter = pm.HalfNormal('jitter', sd=0.01)
    err_scale = pm.HalfNormal('err_scale', sd=1)
    sd = tt.sqrt((err_scale * rv_err) ** 2 + jitter ** 2)
    pm.Normal('y', mu=v_mean, sd=sd, observed=rv )

#ADVI
with my_model:
    advi = pm.ADVI(random_seed=1234)
tracker = pm.callbacks.Tracker(
    mean=advi.approx.mean.eval,  # callable that returns mean
    std=advi.approx.std.eval  # callable that returns std
)
start_time = time.time()  
approx = advi.fit(30000,obj_optimizer=pm.adam(learning_rate=0.01),callbacks=[tracker])
end_time = time.time()
print("time elapsed for advi is : ",end_time-start_time)
print("Loss at the end of training : ",approx.hist[-1])

fig = plt.figure(figsize=(16, 9))
mu_ax = fig.add_subplot(221)
std_ax = fig.add_subplot(222)
hist_ax = fig.add_subplot(212)
mu_ax.plot(tracker['mean'])
mu_ax.set_title('Mean track')
std_ax.plot(tracker['std'])
std_ax.set_title('Std track')
hist_ax.plot(advi.hist)
hist_ax.set_title('Negative ELBO track')
plt.savefig('./plots/convergence.png')
plt.clf()

trace = approx.sample(5000)
print(pm.summary(trace))

with my_model:
    ppc = pm.sample_posterior_predictive(trace,random_seed=1234)
plt.plot(t, rv, '.')
plt.plot(t, ppc['y'][-1], '-')
plt.savefig('./plots/advi.png')
print('Mean squared error of posterior samples and actual data :')
print(((rv - ppc['y'][-1])**2).mean())
plt.clf()
#MCMC
start_time = time.time()
with my_model:
    trace_nuts = pm.sample(draws=5000,random_seed=1234,chains=2,cores=2,init='adapt_diag')
end_time = time.time()
print("time elapsed for MCMC is : ",end_time-start_time)

print(trace_nuts['e'][0:5000].mean())
print(trace_nuts['T'][0:5000].mean())
print(trace_nuts['K'][0:5000].mean())
print(trace_nuts['w'][0:5000].mean())
print(trace_nuts['tau'][0:5000].mean())
print(trace_nuts['v0'][0:5000].mean())

with my_model:
    ppc_nuts = pm.sample_posterior_predictive(trace_nuts,random_seed=1234)

plt.plot(t, rv, '*')
plt.plot(t, ppc_nuts['y'][-1], '-',label='NUTS')
plt.savefig('./plots/nuts.png')
plt.clf()
print('Mean squared error of posterior samples and actual data :')
print(((rv - ppc_nuts['y'][-1])**2).mean())

plt.plot(t[np.arange(0,500,10)], rv[np.arange(0,500,10)], '*',label='data')
plt.plot(t[np.arange(0,500,10)], ppc['y'][-1][np.arange(0,500,10)], '-',label='advi')
plt.plot(t[np.arange(0,500,10)],ppc_nuts['y'][-1][np.arange(0,500,10)],'-',label='nuts')
plt.ylabel('radial velocity (m/s)')
plt.xlabel('time (days)')
plt.legend()
plt.savefig("./plots/final.png")
plt.clf()
plt.figure(figsize=(10,6))

dict_param = {'e':trace['e'],'K':trace['K'],'T':trace['T']}
data_pd = pd.DataFrame.from_dict(dict_param)
g = sns.pairplot(data_pd)
g = g.map_upper(plt.scatter,marker='+')
g = g.map_lower(sns.kdeplot, cmap="hot",shade=True)
g = g.map_diag(sns.kdeplot, shade=True)
plt.savefig('./plots/params.png')
