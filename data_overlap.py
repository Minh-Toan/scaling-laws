import numpy as np
import tensorflow as tf
tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[1], 'GPU')
import hmc
import os

import func

'''
collect data for feature overlaps using HMC sampling
'''

sig = lambda x: tf.nn.tanh(2*x) - 0.72948*x
sig_np = lambda x: np.tanh(2*x) - 0.72948*x
g, dg, l_e, m_e, _, _ = func.sig_related(sig_np)


d = 200
gamma = 0.5
Delta = 0.04
k = int(gamma*d)


def v_sample(vlaw):
    if vlaw == 'beta_03':
        v = func.v_generate(k, beta=0.3)
    if vlaw == 'beta_04':
        v = func.v_generate(k, beta=0.4)
    if vlaw =='beta_07':
        v = func.v_generate(k, beta=0.7)
    if vlaw == 'dense':
        v = np.linspace(2.5, 1, k)
        v /= np.linalg.norm(v)
    if vlaw == 'exp':
        v = np.array([np.exp(-i/20) for i in range(k)])
        v /= np.linalg.norm(v)
    if vlaw == 'exp2':
        v = np.array([np.exp(-i/5) for i in range(k)])
        v /= np.linalg.norm(v)
    return v

alphas = [0.5, 1, 1.5, 2]
vlaws = ['beta_03']


params = {'step_size': 0.01,
          'num_leapfrog_steps': 10,
          'num_adaptation_steps': 2000}

folder=f'data/main_fig/'
os.makedirs(folder, exist_ok=True)

for i in range(16):
    for vlaw in vlaws:
        for alpha in alphas:
            filename = folder+f'{vlaw}_alpha_{alpha}_{i+1}.csv'
            if os.path.exists(filename): # skip the computation below if file already exists
                continue
            v_np = v_sample(vlaw)
            v = tf.constant(v_np, dtype=tf.float32)
            n = int(alpha*d**2)
            # theory
            _, qw_theory = func.solve(n, d, Delta, v_np, g, m_e)
            p = np.argmin(qw_theory>0) # number of learned features
            if p==0: # all features are learned
                p = k

            W0, X, Y = hmc.data_generate(d, k, n, Delta, sig, v)
            top = W0[:p, :]
            bottom = tf.random.normal((k - p, d), dtype=tf.float32)
            W_init = tf.concat([top, bottom], axis=0) # partially informative init

            Ws = hmc.hmc(params, W_init, v, X, Y, Delta, sig)
            Wf = Ws[-1]
            aligns = [func.align(Wf[i], W0[i]) for i in range(k)]
            np.savetxt(filename, aligns, delimiter=',')

