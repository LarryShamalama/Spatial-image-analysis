import cython
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import pymc3 as pm
import os

from theano import scan
import theano.tensor as tt

from pymc3.distributions import continuous
from pymc3.distributions import distribution


N_SAMPLES = 2500 # per chain
N_CHAINS  = 4


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train/255
    x_test  = x_test/255

    O = np.concatenate(x_train[12])

    adj = []
    position_matrix = np.linspace(0, 28*28 - 1, num=28*28).astype(np.int64).reshape(28, 28)
    count = 0

    for i, row in enumerate(position_matrix):
        for j, col in enumerate(position_matrix[i]):
            assert position_matrix[i][j] == col
            
            temp = []

            # change these loops if we do not want to
            # include diagonal elements in adj matrix
            for delta_i in [-1, 0, 1]:
                for delta_j in [-1, 0, 1]:
                    if ((i + delta_i) // 28 == 0) and ((j + delta_j) // 28 == 0):    
                        temp.append(position_matrix[i + delta_i][j + delta_j])
            

            temp.remove(col)
            temp.sort()
            adj.append(temp)
            
    weights = [list(np.ones_like(adj_elems).astype(np.int64)) for adj_elems in adj]

    # below is taken from the pymc3 CAR tutorial website
    maxwz = max([sum(w) for w in weights])
    N = len(weights)
    wmat2 = np.zeros((N, N))
    amat2 = np.zeros((N, N), dtype='int32')
    for i, a in enumerate(adj):
        amat2[i, a] = 1
        wmat2[i, a] = weights[i]


    with pm.Model() as model:
        beta0  = pm.Normal('beta0', mu=0., tau=1e-2)
        tau    = pm.Gamma('tau_c', alpha=1.0, beta=1.0)
        mu_phi = CAR2('mu_phi', w=wmat2, a=amat2, tau=tau, shape=N)
        phi    = pm.Deterministic('phi', mu_phi-tt.mean(mu_phi)) # zero-center phi
        
        mu = pm.Deterministic('mu', beta0 + phi)
        Yi = pm.LogitNormal('Yi', mu=mu, observed=pad(O))
        
        trace = pm.sample(draws=N_SAMPLES, cores=8, tune=500, chains=N_CHAINS)
        posterior_pred = pm.sample_posterior_predictive(trace)

    np.save(new_name(name='mnist_phi_values', suffix='.npy', directory='results/'), trace.get_values('phi'))
