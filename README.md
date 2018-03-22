# Deep Neural Network

## Steps for developing a neural network
  1. Reduce cost
     * gradient reduce
     * ...
  2. Prevent over-fitting
     * more data
     * regularization
     * drop-out
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
  - `W > I` could cause exploding, or vanishing when `W < I`
  ```python
    # init W with zeros will cause all layers do the same thing, cost is not changed (fail to break symmetry)

    # init W with random big numbers can break symmetry, 
    # but it will slow down optimization algorithm (high loss for wrong predict)
    # NOTE: do not initialize parameters too large!

    # *He initialization* works well for networks with ReLU activations.
    W = np.random.randn(n, n-1) * np.sqrt(2 / (n-1)) 
    // Xavier initialization is multiply np.sqrt(1 / (n-1))
  ```

## Normalization
  - Regression
  ```python
    means = X[i].sum() / m
    X = X - means
    
    variances = (np.sum(np.power(X[i], 2)) / m
    X = X / variances
  ```  

  - Batch Norm
  ```python
    # for layer l, Z_l.shape = (l, m)
    means = (Z_i.sum()) / m
    variance = np.sum(np.power(Z_i, 2)) / m
    Z_i_norm = (Z_i -  means) / (np.sqrt(variance + epsilon))
    Z_i_final = gamma * Z_i_norm + beta

  ```

## Cost (gradients)
  ```python
  # cross_entropy_cost
  J(W, b) = - (y * log(a) + (1-y) * log(1-a)) / m

  ```
## Regularization  
  * To reduce/prevent over-fitting.
  * Drive weights to lower values.
  * Hurts training set performance but gives better test accuracy.

  1. L2-regularization 
    - relies on the assumption that *a model with smaller weights is simpler than model with large weights*
    - forward (cost) and backward (W update) propagations both need change
    - weight decay
    
  ```python
    # cost with L2 regularization
    L2_regularization_cost = (lambda / 2m)* (sum(W1^2) + sum(W2^2) + ... + sum(Wn^2)) 
    J_regularization(W, b) = cross_entropy_cost + L2_regularization_cost
    
    # d(L2_regularization_cost) / dW = (lambda / m) * W
    # dW[l] = np.dot(dZ[l], A[l-1].T) / m
    
    dW_regularization[l] = dW[l]+ (lambda / m) * W[l]
    
    # It's called 'weight decay' because this makes weights end up smaller. Why?
    # Because when updating W with dW, (learning_rate * lambda / m) is smaller than 1. 
    # Therefore, updated W[l] is smaller than non-regularization.
    
    W[l] = W[l] - learning_rate * dW_regularization[l]
         = (1 - learning_rate * lambda / m) * W[l] - learning_rate * dW[l]
  ```

  2. Drop-out
    - With dropout, the neurons become less sensitive to the activation. This would prevent overfitting (high variance).
    - Intuition: cannot rely on any one feature, so have to spread out weights (shrink weights).
    - Dropout should be used ONLY in training, not testing.
    - Dropout apply to both forward and backward propagation. 
    - Shown to be an adaptive form without regularization. But drop-out has a similar effect to L2-regularization.
    
  ```python 
    # forward propagation
    D[l] = (np.random.rand(A[l].shape) < keep_prob)
    
    # by dividing keep_prob to keep the same expected value for activations as without dropout (inverted drop-out)
    A[l] = (A[l] * D[l]) / keep_prob  

    # backward propagation
    dA[l] = (dA[l] * D[l]) / keep_prob
  ```

## Gradient Checking
  ```python
  J+ = forward_propagation(W+)
  J- = forward_propagation(W-)

  grad_approx = (J+ - J-) / (2 * epsilon)

  diff = (grad - grad_approx).norm / (grad.norm() + grad_approx.norm())
  ```

## Mini-batch
  - Parameters are adjusted for each epoch X
  * Momentum
    - smooth out the update of parameters in mini-batch
    ```python
    v[dW(i)] = beta * v[dW(i)] + (1-beta)grads[dW(i)]  // default beta = 0.9 is reasonable
    parameters[W(i)] = parameters[W(i)] - learning_rate * v[dW(i)]
    // same for db
    ```
  * Adom
    ```pyton
    v[dW_i] = beta1 * v[dW_i] + (1 - beta1) * dW_i
    v_correct[dW_i] = v[dW_i] / (1 - np.power(beta1, i))

    s[dW_i] = beta2 * s[dW_i] + (1 - beta2) * np.power(dW_i, 2)
    s_correct[dW_i] = s[dW_i] / (1 - np.power(beta2, i))

    W_i = W_i - learning_rate * (v_correct[dW_i] / (np.sqrt(s_correct[dW_i]) + epsilon))
    // same for db
    ```

## Hyper-parameters tuning
  - Scale to pick hyper-parameters
    - linear way is not the best because the impact in different range is quite different
      E.g. beta in momentum, the more beta close to 1, the more impact with tiny changes. (0.9 ~ 0.9005 comparing to 0.999 ~ 0.9995)
  ```python
  # linear r [-4, 0]
  r = -4 * np.random.rand()
  # alpha is in [0.0001, 1]
  alpha = np.power(10, r)

  # liner r [-3, -1]
  r = np.random.uniform(-3, -1)
  # beta is in [0.9, 0.999]
  beta = 1 - np.power(10, r)
  ```

## Convolutional Neural Network
  - convolution operation
    - input * filter  =  output
      - `(n x n) * (f x f) = (n-f+1) x (n-f+1)`
    - problems:
      - vanishing (because output is smaller)
      - unfair to edge pixels 
    - solution: **padding**
      - `p = (f - 1) / 2`
      - output = `(n - f + 2p + 1) x (n - f + 2p + 1)`
    - why filter usually is odd?
      - easy for padding
      - central pixel ???
    - stride
      - output = `((n - f + 2p)/s + 1) x ((n - f + 2p)/s + 1)`
      - `floor()` rounding is used
