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
    image = 'images/malignant_R_MLO.png'

    