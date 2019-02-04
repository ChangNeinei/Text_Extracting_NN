import numpy as np
from util import *
import random
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None
    low = -np.sqrt(6) / np.sqrt(in_size + out_size)
    high = np.sqrt(6) / np.sqrt(in_size + out_size)

    W = np.random.uniform(low, high, in_size * out_size)
    W = W.reshape(in_size, out_size)
    #b = np.random.uniform(low, high, out_size)
    b = np.zeros(out_size)
    
    params['W' + name] = W
    params['b' + name] = b

# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res

# Q 2.2.2
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    # your code here
    pre_act = np.dot(X, W) + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None
    #res = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims = True)
    e_x = np.exp(x - np.max(x))
    sums = np.sum(e_x, axis = 1, keepdims = True)
    res = e_x / sums
    return res

# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None
    loss = -np.sum(y * np.log(probs))
    max_in = np.amax(probs, axis = 1, keepdims = True)
    s = np.ones(probs.shape) * ((probs / max_in) == 1)
    acc = np.sum(y * s) / len(y)
    return loss, acc 

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res


def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name] # [25 * 4] # [2 * 25]
    b = params['b' + name] # [4] # [25]

    X, pre_act, post_act = params['cache_' + name]
    
    data_size = X.shape[0] #40
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X
    # X = [40 * 25] pre_act = [40 * 4]
    error_signal = activation_deriv(post_act) * delta # [40 * 4]
    grad_W = np.dot(X.T, error_signal) #[25 * 4] #[2 * 25]
    grad_b = np.dot(np.ones((data_size)), error_signal) #[4]
    grad_X = (np.dot(W, error_signal.T)).T #[40 * 25]

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 2.4
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    data_size = x.shape[0]
    batch_index = np.random.permutation(data_size)
    batches = []
    for i in range(data_size//int(batch_size)):
        index = batch_index[i * batch_size: i * batch_size + batch_size]
        mini = []
        mini.append(x[index])
        mini.append(y[index])
        batches.append(mini)
    return batches
