# Deep Neural Network


## Activation Functions and Derivatives
* ReLU (Rectified Linear Unit)
  ```
  g(z) = a = max(0, z)
  g'(z) = 0 [when z < 0]; 1 [when z >= 0]
  ```
* Leaky ReLU
  ```
  g(z) = a = max(0.01 * z, z)
  g'(z) = 0.01 [when z < 0]; 1 [when z >= 0]
  ```
* tanh
  ```
  g(z) = a = (e[z] - e[-z]) / (e[z] + e[-z])
  g'(z) = (1 - a.power(2))
  ```  
* sigmoid
  ```
  g(z) = a = 1 / (1 + e.power(-z))
  g'(z) = a * (1 - a)
  ```


## Backward Propagation
  ```
  dA[l] = np.dot(W[l+1].T, dZ[l+1])
  dZ[l] = np.multiply(dA[l], g[l]'(Z[l]))
  dW[l] = np.dot(dZ[l], A[l-1].T) / m
  db[l] = np.sum(dZ[l], axis = 1, keepdims = True) / m

  # dZ[l] = dA[l] * g[l]`(Z[l]) = np.multiply(np.dot(W[l+1].T, dZ[l+1]), g[l]'(Z[l]))
  ```
