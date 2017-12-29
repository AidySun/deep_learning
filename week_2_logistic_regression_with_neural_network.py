def deeplearning(X, Y, lr = 0.005, iter = 1000):
    m = X.shape[1]
    w = np.zeros((X.shape[0], 1))
    b = 0

    for i in range(iter):
        Z = np.dot(W.T, X) + b
        A = sigmoid(Z)
        L = -1 * (np.dot(Y, np.log(A).T) + np.dot((1 - Y), np.log(1 - A).T))
        J = L.sum() / m
        if i % 100 == 0:
            print ("J after iteration %i: %f" %(i, J))
            
        dz = A - Y
        dw = np.dot(X, dz.T) / m
        db = dz.sum() / m

        w = w - np.dot(lr, dw)
        b = b - lr * db

deeplearning(train_set_x, train_set_y)