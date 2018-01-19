# Deep Neural Network

## Steps for developing a neural network
  1. Reduce cost
    * gradient reduce
    * ...
  2. Prevent overfitting
    * more data
    * regularization
    * dorp-out
    * ...

## Basic recipe for ML
  1. High bias (training set performance)
    - larger network
      - more layers
      - more units
    - training longer
    - (NN architecture search)
  2. High variance (dev set performance)
    - more training data
    - regularization
      - L2
      - Dropout
    - (NN architecture search)

## Activation Functions and Derivatives
  * ReLU (Rectified Linear Unit)
    ```python
    g(z) = a = max(0, z)
    g'(z) = 0 [when z < 0]; 1 [when z >= 0]  # (A > 0)
    # dZ = dA * g'(z) = np.multiply(dA, np.int64(A > 0))
    ```
  * Leaky ReLU
    ```python
    g(z) = a = max(0.01 * z, z)
    g'(z) = 0.01 [when z < 0]; 1 [when z >= 0]
    ```
  * tanh
    ```python
    g(z) = a = (e[z] - e[-z]) / (e[z] + e[-z])
    g'(z) = (1 - a.power(2))
    ```  
  * sigmoid
    ```python
    g(z) = a = 1 / (1 + e.power(-z))
    g'(z) = a * (1 - a)
    # da = -(y/a) + (1-y)/(1-a)
    # dz = da * g'(z) = a * (1-a) * (-y/a + (1-y)/(1-a)) = a - y
    ```


## Backward Propagation
  ```python
  dA[l] = np.dot(W[l+1].T, dZ[l+1])
  dZ[l] = np.multiply(dA[l], g[l]'(Z))
  dW[l] = np.dot(dZ[l], A[l-1].T) / m
  db[l] = np.sum(dZ[l], axis = 1, keepdims = True) / m

  # dZ[l] = dA[l] * g[l]'(Z[l]) = np.multiply(np.dot(W[l+1].T, dZ[l+1]), g[l]'(Z[l]))
  ```

## Parameters Initialization
  ```python
    # init W with zeros will cause all layers do the same thing, cost is not changed (faile to break symmetry)
    # init W with big numbers can break semmetry, but it will slow down optimization algorithm (high loss for wrong predict)

    # ReLU should use He Initialization
    W = np.random.randn(n, n-1) * np.sqrt(2 / (n-1)) 
  ```

## Cost
  ```python
  # cross_entropy_cost
  J(W, b) = - (y * log(a) + (1-y) * log(1-a)) / m

  ```
## Regularization  
  * To reduce/prevent overfitting.
  * Drive weights to lower values.
  * Hurts training set performance but gives better test accurancy.

  1. L2-regularization 
    - relies on the assumption that *a model with smaller weights is simpler than model with large weights*

    ```python
    # cost with L2 retularization
    L2_regularization_cost = (lambda / 2m)* (sum(W1^2) + sum(W2^2) + ... + sum(Wn^2)) 
    J_retularization(W, b) = cross_entropy_cost + L2_regularization_cost
  
    # d(L2_regularization_cost) / dW = (lambda / m) * W
    # dW[l] = np.dot(dZ[l], A[l-1].T) / m
  
    dW_regularization[l] = dW[l]+ (lambda / m) * W[l]

    # It's called 'weight decay' because this makes weights end up smaller. Why?
    # Because when updating W with dW, (learning_rate * lambds / m) is smaller than 1. 
    # Therefore, updated W[l] is smaller than non-regularization.

    W[l] = W[l] - learning_rate * dW_regularization[l]
         = (1 - learning_rate * lambda / m) * W[l] - learning_rate * dW[l]
    ```

  2. Drop-out
    - With dropout, the neurons become less sensitive to the activation. This would prevent overfitting (high variance).
    - Intution: cannot rely on any one feature, so have to spread out weights (shrink weights).
    - Dropout should be used ONLY in training, not testing.
    - Dropout applys to both forward and backward propagation. 
    - Shown to be an adaptive form without regularization. But drop-out has a similar effect to L2-regularization.

    ```python
    D[l] = (np.random.rand(A[l].shape) < keep_prob)

    # by deviding keep_prob to keep the same expected value for activations as without dropout (inverted drop-out)
    A[l] = (A[l] * D[l]) / keep_prob  
    ...
    dA[l] = (dA[l] * D[l]) / keep_prob 
  
    ```

## Gradient Checking
  ```python
  J+ = forward_propagation(W+)
  J- = forward_propagation(W-)

  grad_approx = (J+ - J-) / (2 * epsilon)

  diff = (grad - grad_approx).norm / (grad.norm() + grad_approx.norm())
  ```