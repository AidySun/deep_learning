
def neuralNetwork(X, Y, hidden_layers = 4, iterations = 10000, learning_rate = 1.2, print_cost = True):
    n0 = X.shape[0]
    n1 = hidden_layers
    n2 = Y.shape[0] 

    np.random.seed(2)
    W1 = np.random.randn(n1, n0) * 0.01
    b1 = np.zeros((n1, 1))
    W2 = np.random.randn(n2, n1) * 0.01
    b2 = np.zeros((n2, 1))

    m = X.shape[1]
    for i in range(iterations):
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)

        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)

        nplog = np.multiply(Y, np.log(A2)) + np.multiply((1-Y), np.log(1-A2)) 
        J = - nplog.sum() / m
        if print_cost and (i % 1000 == 0):
            print("iteration " + str(i) + " cost is : " + str(np.squeeze(J)))

        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m 

        dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.power(A1, 2)))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

    print("W1 = " + str(W1))
    print("b1 = " + str(b1))
    print("W2 = " + str(W2))
    print("b2 = " + str(b2))
        
#neuralNetwork(X_assess, Y_assess)