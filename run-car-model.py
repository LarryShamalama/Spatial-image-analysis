import cython
import numpy as np
#import matplotlib.pyplot as plt
#import tensorflow as tf -> only needed for loading the mnist dataset
import pymc3 as pm
import pandas as pd
import os


from theano import scan
import theano.tensor as tt

from pymc3.distributions import continuous
from pymc3.distributions import distribution

from lib.car_model import CAR2
from lib.utils import pad, new_name, create_matrices
from matplotlib.image import imread

from argparse import ArgumentParser


N_SAMPLES = 2500 # per chain
N_CHAINS  = 4


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_name',
                        help='Image on which Bayesian model will be fitted on',
                        type=str)

    args = parser.parse_args()


    if args.file_name == 'benign':
        file_name = 'images/benign_R_MLO.jpg'
        image = imread(file_name)
        image = image[:, :, 0]
        image = np.max(image) - image
        print('Analyzing the benign breast cancer image benign_R_MLO.jpg')

    elif args.file_name == 'malignant':
        file_name = 'images/malignant_R_MLO.jpg'
        image = imread(file_name)
        image = image[:, :, 0]
        image = np.max(image) - image
        print('Analyzing the malignant breast cancer image malignant_R_MLO.jpg')

    else:
        # no need to load the entire mnist dataset
        '''
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train/255
        x_test  = x_test/255

        O = np.concatenate(x_train[12])
        '''
        image = pd.read_csv('images/handwritten_digit.csv', header=None).values
        print('Analyzing the handwritten mnist image')


    image = image/np.max(image)
    x_dim, y_dim = image.shape
    pixel_values = np.concatenate(image) # grey scale between 0 and 1
    print('The image is {} by {}\n'.format(x_dim, y_dim))

    N, wmat, amat = create_matrices(x_dim, y_dim)

    with pm.Model() as model:
        beta0  = pm.Normal('beta0', mu=0., tau=1e-2)
        tau    = pm.Gamma('tau_c', alpha=1.0, beta=1.0)
        mu_phi = CAR2('mu_phi', w=wmat, a=amat, tau=tau, shape=N)
        phi    = pm.Deterministic('phi', mu_phi-tt.mean(mu_phi)) # zero-center phi

        mu = pm.Deterministic('mu', beta0 + phi)
        Yi = pm.LogitNormal('Yi', mu=mu, observed=pad(pixel_values))

        trace = pm.sample(draws=N_SAMPLES, cores=8, tune=500, chains=N_CHAINS)
        posterior_pred = pm.sample_posterior_predictive(trace)


    if args.file_name is None:
        prefix_file_name = 'mnist_'
    else:
        prefix_file_name = args.file_name

    np.save(new_name(name=prefix_file_name+'phi_values', suffix='.npy', directory='results/'), trace.get_values('phi'))
