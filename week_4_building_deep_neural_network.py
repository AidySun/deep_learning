
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
    np.random.seed(2)
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

def deep_neural_network(X, Y, layer_dims = [X.shape[0], 4, 1], layer_gs = ['', 'tanh', 'sigmoid'], iterations = 10000, learning_rate = 1.2, print_cost = True):

    parameters = init(layer_dims)
    L = len(layer_dims)

    for iter in range(iterations):
        caches = {} # contains Z and A of each layer, caches['A0'] = X
        caches['A0'] = X

        A_prev = X
        for l in range(1, L):
            A, Z = linear_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], layer_gs[l])
            caches['A' + str(l)] = A
            caches['Z' + str(l)] = Z
            A_prev = A

        J = cost(A, Y)
        if print_cost and iter % 1000 == 0:
            print("cost at iteration %i : %f" %(iter, J))

        dA = -1 * (Y / A) + (1 - Y) / (1 - A)
        for l in range(L - 1, 0, -1):
            dZ, dW, db, dA_prev = linear_backward(dA, caches['A' + str(l-1)], caches['Z' + str(l)], parameters['W' + str(l)], layer_gs[l])
            parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * dW
            parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * db
            dA = dA_prev
    print(parameters)
