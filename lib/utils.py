import numpy as np
import os

'''
Functions and possible objects needed to run Bayesian models
or visualize the results

'''

def pad(array, epsilon=1e-4):
    output = []
    
    for x in array:
        if x == 0:
            output.append(epsilon)
        elif x == 1:
            output.append(1 - epsilon)
        else:
            output.append(x)
            
    return output

def new_name(name, suffix=None, directory='.'):
    assert isinstance(name, str)

    output_name = name

    count = 1

    while (output_name + suffix) in os.listdir(directory):
        if output_name[-2] == '_':
            output_name = output_name[:-2] + '_' + str(count)
        else:
            output_name += '_' + str(count)
        count += 1

    if directory[-1] == '/':
        _directory = directory
    else:
        if directory not in os.listdir():
            os.mkdir(directory)
        _directory = directory + '/'
    
    if suffix is None:
        return _directory + output_name
    else:
        return _directory + output_name + suffix


def expit(n):
    return np.exp(n)/(1 + np.exp(1))