import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!


# Q 2.1
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    t = np.sqrt(6 / (in_size + out_size))
    W = np.random.uniform(-t, t, (in_size, out_size))
    b = np.zeros(out_size)
    
    params['W' + name] = W
    params['b' + name] = b
    return params


# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res


# Q 2.2.1
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
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
    res = np.exp(x - np.max(x, axis=1, keepdims=True))
    return res / np.sum(res, axis=1, keepdims=True)


# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss = np.sum(-np.log(probs) * y, axis=1)
    pred = list(map(lambda x: x == max(x), probs)) * np.ones(shape=probs.shape)

    acc_m = np.multiply(y, pred)
    acc = np.sum(acc_m) / len(acc_m)
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
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    delta *= activation_deriv(post_act)

    grad_X = np.dot(W, delta.T).T
    grad_W = np.dot(delta.T, X)
    grad_b = np.sum(delta.T, 1)

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b

    return grad_X


# Q 2.4
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    idxs = np.arange(len(x))
    np.random.shuffle(idxs)
    batches = []

    lo = 0
    while lo < len(x):
        hi = min(lo + batch_size, len(x))
        b_x = x[lo:hi,:]
        b_y = y[lo:hi,:]
        batches.append((b_x, b_y))
        lo += batch_size

    return batches
