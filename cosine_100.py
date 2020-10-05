import os
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

data1 = np.loadtxt('./crystal2.txt',delimiter=',')
data2 = np.loadtxt('./crystal3.txt',delimiter=',')
data3 = np.loadtxt('./crystal4.txt',delimiter=',')
data4 = np.loadtxt('./crystal6.txt',delimiter=',')
data5 = np.loadtxt('./crystal7.txt',delimiter=',')

data = np.hstack((data1,data2,data3,data4,data5))

def cosine_func(C,p0,p1,A,w,t0,x):
    return C + p0 * np.exp(-np.log(2) * x / p1) + A * np.cos(w * (x - t0))

sigma1=np.array([ np.sqrt( (data1[3,i]**2)) for i in range(len(data1[0])) ])
sigma2=np.array([ np.sqrt( (data2[3,i]**2)) for i in range(len(data2[0])) ])
sigma3=np.array([ np.sqrt( (data3[3,i]**2)) for i in range(len(data3[0])) ])
sigma4=np.array([ np.sqrt( (data4[3,i]**2)) for i in range(len(data4[0])) ])
sigma5=np.array([ np.sqrt( (data5[3,i]**2)) for i in range(len(data5[0])) ])

sigma = np.hstack((sigma1,sigma2,sigma3,sigma4,sigma5))

with pm.Model() as hyp_1 : 
    C1 = pm.Uniform('C1',lower=0.,upper=400.)
    C2 = pm.Uniform('C2',lower=0.,upper=400.)
    C3 = pm.Uniform('C3',lower=0.,upper=400.)
    C4 = pm.Uniform('C4',lower=0.,upper=400.)
    C5 = pm.Uniform('C5',lower=0.,upper=400.)
    p0_1 = pm.Uniform('p0_1',lower=0.,upper=400.)
    p0_2 = pm.Uniform('p0_2',lower=0.,upper=400.)
    p0_3 = pm.Uniform('p0_3',lower=0.,upper=400.)
    p0_4 = pm.Uniform('p0_4',lower=0.,upper=400.)
    p0_5 = pm.Uniform('p0_5',lower=0.,upper=400.)
    p1_1 = pm.Uniform('p1_1',lower=0.,upper=30000.)
    p1_2 = pm.Uniform('p1_2',lower=0.,upper=30000.)
    p1_3 = pm.Uniform('p1_3',lower=0.,upper=30000.)
    p1_4 = pm.Uniform('p1_4',lower=0.,upper=30000.)
    p1_5 = pm.Uniform('p1_5',lower=0.,upper=30000.)
    A = pm.Uniform('A',lower=0.,upper=400.)
    t0 = pm.Uniform('t0',lower=0.,upper=360.)
    w = pm.Uniform('w',lower=0.0104,upper=0.428)
    mean1 = cosine_func(C1,p0_1,p1_1,A,w,t0,data1[0])
    mean2 = cosine_func(C2,p0_2,p1_2,A,w,t0,data2[0])
    mean3 = cosine_func(C3,p0_3,p1_3,A,w,t0,data3[0])
    mean4 = cosine_func(C4,p0_4,p1_4,A,w,t0,data4[0])
    mean5 = cosine_func(C5,p0_5,p1_5,A,w,t0,data5[0])
    mean = pm.math.concatenate([mean1,mean2,mean3,mean4,mean5])
    pm.Normal('y',mu=mean,sd=sigma,observed=data[1])

with hyp_1:
    advi = pm.ADVI(random_state=1234)
tracker = pm.callbacks.Tracker(
    mean=advi.approx.mean.eval,  # callable that returns mean
    std=advi.approx.std.eval  # callable that returns std
)

approx = advi.fit(1500000,obj_optimizer=pm.adam(learning_rate=5e-5),callbacks=[pm.callbacks.CheckParametersConvergence(diff='absolute'), tracker])

# convergence check
# fig = plt.figure(figsize=(16, 9))
# mu_ax = fig.add_subplot(221)
# std_ax = fig.add_subplot(222)
# hist_ax = fig.add_subplot(212)
# mu_ax.plot(tracker['mean'])
# mu_ax.set_title('Mean track')
# std_ax.plot(tracker['std'])
# std_ax.set_title('Std track')
# hist_ax.plot(advi.hist)
# hist_ax.set_title('Negative ELBO track')
# plt.savefig('cosine_converg.png')

trace = approx.sample(500)

#get parameter values
# df = pm.summary(trace).reset_index()
# df.to_csv('hyp1.csv')
# h1_c1 = df[df['index'] == 'C1']['mean'].item()
# h1_c2 = df[df['index'] == 'C2']['mean'].item()
# h1_c3 = df[df['index'] == 'C3']['mean'].item()
# h1_c4 = df[df['index'] == 'C4']['mean'].item()
# h1_c5 = df[df['index'] == 'C5']['mean'].item()

# h1_p0_1 = df[df['index'] == 'p0_1']['mean'].item()
# h1_p0_2 = df[df['index'] == 'p0_2']['mean'].item()
# h1_p0_3 = df[df['index'] == 'p0_3']['mean'].item()
# h1_p0_4 = df[df['index'] == 'p0_4']['mean'].item()
# h1_p0_5 = df[df['index'] == 'p0_5']['mean'].item()

# h1_p1_1 = df[df['index'] == 'p1_1']['mean'].item()
# h1_p1_2 = df[df['index'] == 'p1_2']['mean'].item()
# h1_p1_3 = df[df['index'] == 'p1_3']['mean'].item()
# h1_p1_4 = df[df['index'] == 'p1_4']['mean'].item()
# h1_p1_5 = df[df['index'] == 'p1_5']['mean'].item()

# h1_a = df[df['index'] == 'A']['mean'].item()
# h1_t0 = df[df['index'] == 't0']['mean'].item()
# h1_w = df[df['index'] == 'w']['mean'].item()



with pm.Model() as hyp_2 : 
    C1 = pm.Uniform('C1',lower=0.,upper=400.)
    C2 = pm.Uniform('C2',lower=0.,upper=400.)
    C3 = pm.Uniform('C3',lower=0.,upper=400.)
    C4 = pm.Uniform('C4',lower=0.,upper=400.)
    C5 = pm.Uniform('C5',lower=0.,upper=400.)
    p0_1 = pm.Uniform('p0_1',lower=0.,upper=400.)
    p0_2 = pm.Uniform('p0_2',lower=0.,upper=400.)
    p0_3 = pm.Uniform('p0_3',lower=0.,upper=400.)
    p0_4 = pm.Uniform('p0_4',lower=0.,upper=400.)
    p0_5 = pm.Uniform('p0_5',lower=0.,upper=400.)
    p1_1 = pm.Uniform('p1_1',lower=0.,upper=30000.)
    p1_2 = pm.Uniform('p1_2',lower=0.,upper=30000.)
    p1_3 = pm.Uniform('p1_3',lower=0.,upper=30000.)
    p1_4 = pm.Uniform('p1_4',lower=0.,upper=30000.)
    p1_5 = pm.Uniform('p1_5',lower=0.,upper=30000.)
    
    mean1 = cosine_func(C1,p0_1,p1_1,0,0,0,data1[0])
    mean2 = cosine_func(C2,p0_2,p1_2,0,0,0,data2[0])
    mean3 = cosine_func(C3,p0_3,p1_3,0,0,0,data3[0])
    mean4 = cosine_func(C4,p0_4,p1_4,0,0,0,data4[0])
    mean5 = cosine_func(C5,p0_5,p1_5,0,0,0,data5[0])
    mean = pm.math.concatenate([mean1,mean2,mean3,mean4,mean5])
    pm.Normal('y',mu=mean,sd=sigma,observed=data[1])


with hyp_2:
    advi_2 = pm.ADVI(random_state=1234)
tracker_2 = pm.callbacks.Tracker(
    mean=advi_2.approx.mean.eval,  # callable that returns mean
    std=advi_2.approx.std.eval  # callable that returns std
)

approx_2 = advi_2.fit(1500000,obj_optimizer=pm.adam(learning_rate=5e-5),callbacks=[pm.callbacks.CheckParametersConvergence(diff='absolute'), tracker_2])

#convergence check
# fig = plt.figure(figsize=(16, 9))
# mu_ax = fig.add_subplot(221)
# std_ax = fig.add_subplot(222)
# hist_ax = fig.add_subplot(212)
# mu_ax.plot(tracker_2['mean'])
# mu_ax.set_title('Mean track')
# std_ax.plot(tracker_2['std'])
# std_ax.set_title('Std track')
# hist_ax.plot(advi_2.hist)
# hist_ax.set_title('Negative ELBO track')
# plt.savefig('cosine_converg_h2.png')

trace_2 = approx_2.sample(500)

# get parameter values
# df = pm.summary(trace_2).reset_index()
# df.to_csv('hyp2.csv')
# h2_c1 = df[df['index'] == 'C1']['mean'].item()
# h2_c2 = df[df['index'] == 'C2']['mean'].item()
# h2_c3 = df[df['index'] == 'C3']['mean'].item()
# h2_c4 = df[df['index'] == 'C4']['mean'].item()
# h2_c5 = df[df['index'] == 'C5']['mean'].item()

# h2_p0_1 = df[df['index'] == 'p0_1']['mean'].item()
# h2_p0_2 = df[df['index'] == 'p0_2']['mean'].item()
# h2_p0_3 = df[df['index'] == 'p0_3']['mean'].item()
# h2_p0_4 = df[df['index'] == 'p0_4']['mean'].item()
# h2_p0_5 = df[df['index'] == 'p0_5']['mean'].item()

# h2_p1_1 = df[df['index'] == 'p1_1']['mean'].item()
# h2_p1_2 = df[df['index'] == 'p1_2']['mean'].item()
# h2_p1_3 = df[df['index'] == 'p1_3']['mean'].item()
# h2_p1_4 = df[df['index'] == 'p1_4']['mean'].item()
# h2_p1_5 = df[df['index'] == 'p1_5']['mean'].item()

# data fit plot using the paramters obtained from ADVI

# plt.figure(figsize=(12,15))
# plt.subplot(5,1,1)
# plt.errorbar(x=data1[0],y=data1[1],yerr=data1[3],fmt='.',c='k')
# plt.plot(data1[0],cosine_func(h1_c1,h1_p0_1,h1_p1_1,h1_a,h1_w,h1_t0,data1[0]),c='r',label='H1')
# plt.plot(data1[0],cosine_func(h2_c1,h2_p0_1,h2_p1_1,0,0,0,data1[0]),'c-.',label='H2')
# plt.title('crystal2')
# plt.legend()
# plt.subplot(5,1,2)
# plt.errorbar(x=data2[0],y=data2[1],yerr=data2[3],fmt='.',c='k')
# plt.plot(data2[0],cosine_func(h1_c2,h1_p0_2,h1_p1_2,h1_a,h1_w,h1_t0,data2[0]),c='r',label='H1')
# plt.plot(data2[0],cosine_func(h2_c2,h2_p0_2,h2_p1_2,0,0,0,data2[0]),'c-.',label='H2')
# plt.title('crystal3')
# plt.legend()
# plt.subplot(5,1,3)
# plt.errorbar(x=data3[0],y=data3[1],yerr=data3[3],fmt='.',c='k')
# plt.plot(data3[0],cosine_func(h1_c3,h1_p0_3,h1_p1_3,h1_a,h1_w,h1_t0,data3[0]),c='r',label='H1')
# plt.plot(data3[0],cosine_func(h2_c3,h2_p0_3,h2_p1_3,0,0,0,data3[0]),'c-.',label='H2')
# plt.title('crystal4')
# plt.legend()
# plt.subplot(5,1,4)
# plt.errorbar(x=data4[0],y=data4[1],yerr=data4[3],fmt='.',c='k')
# plt.plot(data4[0],cosine_func(h1_c4,h1_p0_4,h1_p1_4,h1_a,h1_w,h1_t0,data4[0]),c='r',label='H1')
# plt.plot(data4[0],cosine_func(h2_c4,h2_p0_4,h2_p1_4,0,0,0,data4[0]),'c-.',label='H2')
# plt.title('crystal6')
# plt.legend()
# plt.subplot(5,1,5)
# plt.errorbar(x=data5[0],y=data5[1],yerr=data5[3],fmt='.',c='k')
# plt.plot(data5[0],cosine_func(h1_c5,h1_p0_5,h1_p1_5,h1_a,h1_w,h1_t0,data5[0]),c='r',label='H1')
# plt.plot(data5[0],cosine_func(h2_c5,h2_p0_5,h2_p1_5,0,0,0,data5[0]),'c-.',label='H2')
# plt.title('crystal7')
# plt.legend()
# plt.tight_layout()
# plt.show()
# plt.savefig('advi_plot.png')

#calculating ELBO
Z1 = np.exp(np.array([hyp_1.logp(pt) for pt in pm.sample_approx(approx)])).mean()
print(np.log(Z1))

Z2 = np.exp(np.array([hyp_2.logp(pt) for pt in pm.sample_approx(approx_2)])).mean()
print(np.log(Z2))