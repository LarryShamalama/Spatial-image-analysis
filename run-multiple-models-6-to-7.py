import cython
import numpy as np
import tensorflow as tf
import pymc3 as pm
#import pandas as pd
import os, time


from theano import scan
import theano.tensor as tt

from pymc3.distributions import continuous
from pymc3.distributions import distribution

from lib.car_model import CAR2
from lib.utils import pad, new_name, create_matrices, get_digit_indices
from matplotlib.image import imread

#from argparse import ArgumentParser

N_SAMPLES = 1500 # per chain
N_CHAINS  = 2
N_TUNE    = 300

DIRECTORY = 'results/multiple-analyses/'
QUANTITY  = 10



if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train/255
    x_test  = x_test/255


    for i in [6, 7]:
        start = time.time()
        print('Digit ' + str(i))
        print('\n')
        labels = get_digit_indices(y_train, i, QUANTITY)

        for label in labels:
            image = x_train[label]
            x_dim, y_dim = image.shape
            pixel_values = np.concatenate(image) # grey scale between 0 and 1

            N, wmat, amat = create_matrices(x_dim, y_dim)

            with pm.Model() as model:
                beta0  = pm.Normal('beta0', mu=0., tau=1e-2)
                tau    = pm.Gamma('tau_c', alpha=1.0, beta=1.0)
                mu_phi = CAR2('mu_phi', w=wmat, a=amat, tau=tau, shape=N)
                phi    = pm.Deterministic('phi', mu_phi-tt.mean(mu_phi)) # zero-center phi

                mu = pm.Deterministic('mu', beta0 + phi)
                Yi = pm.LogitNormal('Yi', mu=mu, observed=pad(pixel_values))

                max_a_post = pm.find_MAP()
                step  = pm.NUTS()
                trace = pm.sample(draws=N_SAMPLES, step=step, start=max_a_post, cores=2, tune=N_TUNE, chains=N_CHAINS)
                posterior_pred = pm.sample_posterior_predictive(trace)

                prefix_file_name = 'mnist_digit{}(label{})_'.format(i, label)

                np.save(new_name(name=prefix_file_name+'phi_values', suffix='.npy', directory=DIRECTORY), trace.get_values('phi'))
        
        print('Finished fitting models on digit ' + str(i))
        print('Took {} seconds'.format(np.around(time.time() - start, 0)))
        print('\n')
