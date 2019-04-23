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


def create_matrices(x_dim, y_dim):
    '''
    output are inputs for CAR2 object

    matrices are running into memory issues though...
    '''
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
                    if ((i + delta_i) // x_dim == 0) and ((j + delta_j) // y_dim == 0):    
                        temp.append(position_matrix[i + delta_i][j + delta_j])
            
            try:
                temp.remove(col)
            except Exception as e:
                print('temp: ', temp)
                print('col: ', col)
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

    return N, wmat, amat


def get_digit_indices(y_train, digit, quantity):
    '''
    obtains the index of a given digit within the x_train, y_train
    '''
    assert 0 <= digit and digit <= 9
    assert quantity > 0


    return np.argwhere(y_train == digit).reshape(-1,)[:quantity]