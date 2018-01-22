
"""

* mini-batch
* bias correction
* improved gradient descent
* adam

"""

"""
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
"""
%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#################### Helpers ######################

def relu_func(x):    
    return np.maximum(0, x)

def relu_derivative_func(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ


def tanh_func(x):    
    return np.tanh(x)

def tanh_derivative_func(dA, Z): 
    A = tanh_func(Z)
    dZ = np.multiply(dA, (1 - np.power(A, 2)))
    return dZ


def sigmoid_func(x):
    return (1 / (1 + np.exp(-x)))

def sigmoid_derivative_func(dA, Z):
    A = sigmoid_func(Z)
    dZ = dA * A * (1 - A)
    return dZ
     
#################### Business ######################

def init(layer_dims):
    """ 
    Arguments:
    layer_dims: list contains the dimonsion number of layers (starts from 0 [feature number] to L)

    Returns:
    parameters: dictionary contains W and b of each layer (index examples: "W1", "b3")
    """ 
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)

    for i in range(1, L):
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))
    return parameters

def A_by_func(Z, activation_func):
    if activation_func == 'sigmoid':
        A = sigmoid_func(Z)
    elif activation_func == 'relu':
        A = relu_func(Z)
    elif activation_func == 'tanh':
        A = tanh_func(Z)
    assert(Z.shape == A.shape)
    return A

def linear_forward(A_prev, W, b, activation_func):
    Z = np.dot(W, A_prev) + b
    A = A_by_func(Z, activation_func)
    assert(A.shape == Z.shape)
    return A, Z

def model_forward_propagation(X, parameters, layer_functions):
    L = len(layer_functions)

    caches = {}
    A_prev = X
    for l in range(1, L):
        A, Z = linear_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], layer_functions[l])
        caches['A' + str(l)] = A
        caches['Z' + str(l)] = Z
        A_prev = A
    return caches, A


def cost(AL, Y):
    m = Y.shape[1]
    J = - (np.dot(Y, np.log(AL.T)) + np.dot((1 - Y), np.log(1 - AL.T))).sum() / m
    J = np.squeeze(J)
    return J

def dZ_by_func(dA, Z, activation_func):
    if activation_func == 'sigmoid':
        dZ = sigmoid_derivative_func(dA, Z)
    elif activation_func == 'relu':
        dZ = relu_derivative_func(dA, Z)
    elif activation_func == 'tanh':
        dZ = tanh_derivative_func(dA, Z)
    assert(dZ.shape == Z.shape)
    return dZ

def linear_backward(dA, A_prev, Z, W, activation_func):
    dZ = dZ_by_func(dA, Z, activation_func)

    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis = 1, keepdims = True) / m
    dA_prev = np.dot(W.T, dZ)

    return dZ, dW, db, dA_prev

def model_backward_propagation(AL, Y, parameters, caches, layer_gs):
    dA = -1 * (Y / AL) + (1 - Y) / (1 - AL) # assume last layer is sigmoid 

    L = len(layer_gs)

    grads = {}
    for l in range(L - 1, 0, -1):
        dZ, dW, db, dA_prev = linear_backward(dA, caches['A' + str(l-1)], caches['Z' + str(l)], parameters['W' + str(l)], layer_gs[l])
        grads['dW' + str(l)] = dW
        grads['db' + str(l)] = db
        grads['dZ' + str(l)] = dZ
        dA = dA_prev
    return grads
  
def update_parameters(parameters, grads, learning_rate, L):
    for l in range(1, L):
        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * grads['db' + str(l)]
    return parameters

def update_parameters_adam(parameters, adam_parameters, grads, learning_rate, L, beta1, beta2, epsilon):
    for l in range(1, L):
        n = str(l)
        dW = grads['dW' + n]
        db = grads['db' + n]
        adam_parameters["vdW" + n] = beta1 * adam_parameters["vdW" + n] + (1 - beta1)*dW
        adam_parameters["vdb" + n] = beta1 * adam_parameters["vdb" + n] + (1 - beta1)*db
        adam_parameters["sdW" + n] = beta2 * adam_parameters["sdW" + n] + (1 - beta2)*(np.power(dW, 2))
        adam_parameters["sdb" + n] = beta2 * adam_parameters["sdb" + n] + (1 - beta2)*(np.power(db, 2))

        # bias correction
        vdW_corrected = adam_parameters["vdW" + n] / (1 - np.power(beta1, l))
        vdb_corrected = adam_parameters["vdb" + n] / (1 - np.power(beta1, l))
        sdW_corrected = adam_parameters["sdW" + n] / (1 - np.power(beta2, l))
        sdb_corrected = adam_parameters["sdb" + n] / (1 - np.power(beta2, l))

        parameters['W' + n] = parameters['W' + n] - learning_rate * (vdW_corrected / (np.sqrt(sdW_corrected) + epsilon))
        parameters['b' + n] = parameters['b' + n] - learning_rate * (vdb_corrected / (np.sqrt(sdb_corrected) + epsilon))
    return parameters, adam_parameters

def initialize_adam_parameters(layer_dims):
    L = len(layer_dims)

    adam_parameters = {}
    for l in range(1, L):
        n = str(l)
        adam_parameters["vdW" + n] = np.zeros((layer_dims[l], layer_dims[l - 1]))
        adam_parameters["vdb" + n] = np.zeros((layer_dims[l], 1))
        adam_parameters["sdW" + n] = np.zeros((layer_dims[l], layer_dims[l - 1]))
        adam_parameters["sdb" + n] = np.zeros((layer_dims[l], 1))
    return adam_parameters

def split(X, Y, mini_batch_num, seed):
    m = X.shape[1]

    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation].reshape((1,m))

    count = math.floor(m / mini_batch_num)

    X_splitted = {}
    Y_splitted = {}

    for i in range(count):
        X_splitted["X" + str(i)] = X_shuffled[:, i * mini_batch_num : (i+1) * mini_batch_num]
        Y_splitted["Y" + str(i)] = Y_shuffled[:, i * mini_batch_num : (i+1) * mini_batch_num]

    if m % mini_batch_num != 0:
        X_splitted["X" + str(count)] = X_shuffled[:, count * mini_batch_num : m]
        Y_splitted["Y" + str(count)] = Y_shuffled[:, count * mini_batch_num : m]
        count += 1
    return X_splitted, Y_splitted, count


#def deep_neural_network(X, Y, layer_dims, layer_gs, iterations = 10000, learning_rate = 1.2, print_cost = True):
    ## call optimized
    #deep_neural_network_optimized(X, Y, layer_dims, layer_gs, iterations, learning_rate, mini_batch_num = X.shape[1], beta1 = 0, beta2 = 0, epsilon = 0, print_cost)

def deep_neural_network_optimized(X, Y, layer_dims, layer_gs, iterations = 10000, learning_rate = 1.2, mini_batch_num = 64, beta1 = 0.9, beta2 = 0.999, epsilon = 10e-8, print_cost = True):

    print("X.shape = " + str(X.shape))
    print("Y.shape = " + str(Y.shape))
    print("layer_dims = " + str(layer_dims))
    print("layer_gs = " + str(layer_gs))
    print("iterations = " + str(iterations))
    print("learning_rate = " + str(learning_rate))

    # *NOTE*: init(layer_dims) in notebook may get diff result comparing with running 
    # initialize_parameters_deep() on server, even with same seed.
    parameters = initialize_parameters(layer_dims) # week 4 : initialize_parameters_deep(layer_dims)
    adam_parameters = initialize_adam_parameters(layer_dims)
    L = len(layer_dims)


    costs = []
    for iter in range(iterations):
        X_splitted, Y_splitted, mini_batch_count = split(X, Y, mini_batch_num, iter + 1)

        for batch_i in range(mini_batch_count):
            Xi = X_splitted["X" + str(batch_i)]
            Yi = Y_splitted["Y" + str(batch_i)]

            caches = {} # contains Z and A of each layer, caches['A0'] = X
            caches['A0'] = Xi

            forward_caches, AL = model_forward_propagation(Xi, parameters, layer_gs)
            caches.update(forward_caches)

            J = cost(AL, Yi)
            
            grads = model_backward_propagation(AL, Yi, parameters, caches, layer_gs)
            parameters, adam_parameters = update_parameters_adam(parameters, adam_parameters, grads, learning_rate, L, beta1, beta2, epsilon)

        if print_cost and iter % 1000 == 0:
            print("cost at iteration %i : %f" %(iter, J))
            costs.append(J)           

    # print(parameters)
    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters

"""
layer_dims = [train_X.shape[0], 5, 2, 1]
parameters = deep_neural_network_optimized(train_X, train_Y, layer_dims, layer_gs = ['', 'relu', 'relu', 'sigmoid'], iterations = 10000, learning_rate = 0.0007, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, print_cost = True)


# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

"""
