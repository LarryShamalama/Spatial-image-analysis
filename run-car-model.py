import cython
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import pymc3 as pm
import pandas as pd
import os


from theano import scan
import theano.tensor as tt

from pymc3.distributions import continuous
from pymc3.distributions import distribution

from lib.car_model import CAR2
from lib.utils import pad, new_name
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
        image /= np.max(image)
        print('Analyzing the benign breast cancer image benign_R_MLO.jpg')

    elif args.file_name == 'malignant':
        file_name = 'images/malignant_R_MLO.jpg'
        image = imread(file_name)
        image = image[:, :, 0]
        image = np.max(image) - image
        image /= np.max(image)
        print('Analyzing the benign breast cancer image malignant_R_MLO.jpg')

    else:
        # no need to load the entire mnist dataset
        '''
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train/255
        x_test  = x_test/255

        O = np.concatenate(x_train[12])
        '''
        image = pd.read_csv('images/handwritten_digit.csv').values
        image /= max(image)
        print('Analyzing the handwritten mnist image')
    


    x_dim, y_dim = image.shape
    print('The image is {} by {}'.format(x_dim, y_dim))

    adj = []
    position_matrix = np.linspace(0, x_dim*y_dim - 1, num=x_dim*y_dim).astype(np.int64).reshape(x_dim, y_dim)
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
    wmat = np.zeros((N, N))
    amat = np.zeros((N, N), dtype='int32')
    for i, a in enumerate(adj):
        amat[i, a] = 1
        wmat[i, a] = weights[i]


    with pm.Model() as model:
        beta0  = pm.Normal('beta0', mu=0., tau=1e-2)
        tau    = pm.Gamma('tau_c', alpha=1.0, beta=1.0)
        mu_phi = CAR2('mu_phi', w=wmat, a=amat, tau=tau, shape=N)
        phi    = pm.Deterministic('phi', mu_phi-tt.mean(mu_phi)) # zero-center phi
        
        mu = pm.Deterministic('mu', beta0 + phi)
        Yi = pm.LogitNormal('Yi', mu=mu, observed=pad(O))
        
        trace = pm.sample(draws=N_SAMPLES, cores=8, tune=500, chains=N_CHAINS)
        posterior_pred = pm.sample_posterior_predictive(trace)


    if args.file_name is None:
        prefix_file_name = 'mnist_'
    else:
        prefix_file_name = args.file_name

    np.save(new_name(name=prefix_file_name+'phi_values', suffix='.npy', directory='results/'), trace.get_values('phi'))