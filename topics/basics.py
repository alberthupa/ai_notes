import numpy as np

# Initialize variables
X = np.random.rand(3, 3)
y = np.random.rand(3, 1)
w = np.zeros((3, 1))
alpha = 0.01  # learning rate

# Compute the gradient
grad = (2 / 3) * X.T @ (X @ w - y)


# Update weights
w_new = w - alpha * grad

w_new
"""
The expression `grad = (2 / 3) * X.T @ (X @ w - y)` calculates the gradient of the mean squared error cost function with respect to the weights \(w\). This step is crucial in the gradient descent algorithm as it guides how we adjust the weights to minimize the cost function. Let's break it down:

1. **Prediction**: \(X @ w\)
   - This part computes the predicted output by multiplying the input matrix \(X\) with the weight vector \(w\). The result is a vector of predictions corresponding to the inputs.

2. **Compute Error**: \(X @ w - y\)
   - After obtaining the predictions, we subtract the actual outputs \(y\) from these predictions. The result is the error or residual for each observation. It indicates how far off our predictions are from the actual values.

3. **Scale and Prepare for Gradient**: \(X^T @ (X @ w - y)\)
   - We then multiply the error by the transpose of the input matrix \(X^T\). This operation is part of computing the gradient of the cost function. It scales the errors by the input values, essentially determining the contribution of each feature's error to the gradient.
   - The transpose of \(X\) is used to ensure the matrix dimensions align for multiplication, and to properly aggregate the contributions of each error to the respective weights.

4. **Average and Scale Gradient**: \(\frac{2}{n} X^T @ (X @ w - y)\)
   - The factor \(\frac{2}{n}\) is applied to the result of the previous step. Here, \(n\) is the number of observations (3 in this case).
   - The factor of 2 comes from the derivative of the squared term in the cost function, and the division by \(n\) averages the gradient across all observations. This makes the gradient a reflection of the average error slope with respect to each weight.

5. **Store in `grad`**:
   - Finally, the computed gradient is stored in the variable `grad`. This gradient points in the direction of the steepest increase of the cost function. However, in gradient descent, we intend to decrease the cost, so we will move the weights in the opposite direction of this gradient.

In summary, this step efficiently computes how much and in what direction the weights \(w\) should be adjusted to reduce the mean squared error across all observations, thereby improving the model's predictions.
"""