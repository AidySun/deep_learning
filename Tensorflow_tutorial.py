# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


def W(i):
    return "W" + str(i)

def b(i):
    return "b" + str(i)

def init_parameters(layer_dims):
    """
    Arguments: 
    layer_dims: list contains hidden unit number of each layer (including the input layer at index 0)

    Returns:
    parameters: dictionary of parameters with key "Wx" and "bx" (e.g. "W1", "b3")
    """

    # layer numbers including input layer
    L = len(layer_dims)

    parameters = {}

    for i in range(1, L):
        parameters[W(i)] = tf.get_variable(
         name = W(i),
         shape = [layer_dims[i], layer_dims[i-1]],
         initializer = tf.contrib.layers.xavier_initializer(seed = 1))

        parameters[b(i)] = tf.get_variable(
           name = b(i),
           shape = [layer_dims[i], 1],
           initializer = tf.zeros_initializer())

    assert(len(parameters) == (2 * (L-1)))
    return parameters

def forward_propagation(X, layer_dims, parameters):
    """
    Assuming:
     - model is Relu->Relu-> ... ->Softmax
     - forward propagation computes all Relu layers but final Softmax layer

    Returns:
    ZL : Z of last Relu layer
    """
    L = len(layer_dims)

    A_pre = tf.cast(X, tf.float32)

    for l in range(1, L):
        Zl = tf.matmul(parameters[W(l)], A_pre) + parameters[b(l)]
        Al = tf.nn.relu(Zl)
        A_pre = Al

    return Zl

def calculate_cost(ZL, Y):
    logits = tf.transpose(ZL)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    return cost


def split(matrix, by):
    m = matrix.shape[1]

    count = math.floor(m / by)

    splitted = []
    for i in range(count):
        splitted.append(matrix[:, i * by : (i + 1) * by])

    if m % by != 0:
        splitted.append(matrix[:, count * by : m])

    return splitted

def shuffle_and_split_data(X, Y, minibatch_size, seed):
    np.random.seed(seed)

    m = X.shape[1]
    permutation = list(np.random.permutation(m))

    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    splitted_X = split(shuffled_X, minibatch_size)
    splitted_Y = split(shuffled_Y, minibatch_size)

    return (splitted_X, splitted_Y)


def tensorflow_model(X_train, 
                     Y_train,
                     X_test,
                     Y_test,
                     layer_dims,  # E.g. [1024, 12, 4, 6] is 3 hidden layers with 1024 features (X_train.shape[0])
                     learning_rate = 0.0001,
                     iterations = 1500,
                     minibatch_size = 32):
    """
    Model with Relu->Relu->...->Softmax.
    """
    print("X_train.shape = " + str(X_train.shape))
    print("Y_train.shape = " + str(Y_train.shape))
    print("X_test.shape = " + str(X_test.shape))
    print("Y_test.shape = " + str(Y_test.shape))
    print("layer_dims = " + str(layer_dims))
    print("learning_rate = " + str(learning_rate))
    print("iterations = " + str(iterations))
    print("minibatch_size = " + str(minibatch_size))
    print("\n")
                        
    tf.reset_default_graph()
    tf.set_random_seed(1)

    (features, m) = X_train.shape 
    classification = Y_train.shape[0]  
    
    X = tf.placeholder(tf.float32, shape = (features, None))
    Y = tf.placeholder(tf.float32, shape = (classification, None))

    parameters = init_parameters(layer_dims)
    ZL = forward_propagation(X, layer_dims, parameters)
    cost = calculate_cost(ZL, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    costs = []
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)

        for i in range(iterations + 1):
            (splitted_X, splitted_Y) = shuffle_and_split_data(X_train, Y_train, minibatch_size, i + 1)

            num_minibatches = len(splitted_X)

            epoch_cost = 0.
            for j in range(num_minibatches):
                _ , minibatch_cost = session.run([optimizer, cost], feed_dict = {X: splitted_X[j], Y: splitted_Y[j]})
                epoch_cost += minibatch_cost / num_minibatches

            if i % 200 == 0:
                print("cost after %i iteration is : %f" % (i, epoch_cost))
            if i % 5 == 0 :
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = session.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        #session.close()

    return parameters

"""
layer_dims = [X_train.shape[0], 25, 12, 6]
parameters = tensorflow_model(X_train, Y_train, X_test, Y_test, layer_dims = layer_dims)
"""