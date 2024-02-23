import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Backpropagation - going back through the network and adjusting the weights and biases to minimize the loss function. It's based on the gradient descent optimization algorithm, where the gradient of the loss function with respect to each weight is computed, and the weights are updated in the direction that reduces the loss.

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid (for backpropagation)
def sigmoid_derivative(x):
    return x * (1 - x)

# Generating random data
np.random.seed(42) # For reproducibility
input_data = np.random.rand(100, 3) # 100 samples, 3 features
real_output = np.random.rand(100, 1) # 100 samples, 1 output

# Neural Network parameters
inputLayer_neurons = input_data.shape[1] # number of features in data
hiddenLayer_neurons = 4 # number of hidden layers neurons
outputLayer_neurons = 1 # number of neurons at output layer

# Weight and bias initialization
hidden_weights = np.random.uniform(size=(inputLayer_neurons, hiddenLayer_neurons))
hidden_bias =np.random.uniform(size=(1, hiddenLayer_neurons))
output_weights = np.random.uniform(size=(hiddenLayer_neurons,outputLayer_neurons)) # o ile trzeba zmienic zeby bylo dobrze
output_bias = np.random.uniform(size=(1,outputLayer_neurons)) # o ile trzeba zmienic zeby bylo dobrze

# Learning rate
lr = 0.1

# Initialize list to store mean squared error
mse_history = []

# Training the network
for _ in range(100000):
    # Forward Propagation
    # Compute input for the hidden layer by multiplying input data with hidden weights and adding hidden bias.
    hidden_layer_input = np.dot(input_data, hidden_weights) # Dot product of input data and hidden layer's weights
    hidden_layer_input += hidden_bias # Adding bias to each neuron in the hidden layer
    
    # Apply activation function (sigmoid) to the hidden layer input to get hidden layer activations.
    hidden_layer_activation = sigmoid(hidden_layer_input) # Activation function adds non-linearity
    
    # Compute input for the output layer by multiplying hidden layer activations with output weights and adding output bias.
    output_layer_input = np.dot(hidden_layer_activation, output_weights) # Dot product of hidden layer activations and output layer's weights
    output_layer_input += output_bias # Adding bias to each neuron in the output layer
    
    # Apply activation function (sigmoid) to the output layer input to get final predictions.
    predicted_output = sigmoid(output_layer_input) # Final output prediction
    
    # Backpropagation
    # Calculate the difference between actual and predicted output (error).
    error = real_output - predicted_output # Error in prediction
    
    # Calculate the derivative of the predicted output (gradient of the loss w.r.t. output predictions).
    d_predicted_output = error * sigmoid_derivative(predicted_output) # Element-wise multiplication for error gradient
    
    # Calculate error for the hidden layer by multiplying the derivatives of predicted output with the output weights.
    error_hidden_layer = d_predicted_output.dot(output_weights.T) # Error propagated back to the hidden layer
    
    # Calculate the derivative of the hidden layer output (gradient of the loss w.r.t. hidden layer activations).
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activation) # Element-wise multiplication for hidden layer error gradient
    
    # Updating Weights and Biases
    # Update output weights by multiplying the transpose of hidden layer activations with the derivative of predicted output.
    # Learning rate is used to control how much weights are updated during each iteration.
    output_weights += hidden_layer_activation.T.dot(d_predicted_output) * lr # Update output layer weights
    
    # Update output bias by summing the derivatives of the predicted output and multiplying by the learning rate.
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr # Update output layer bias
    
    # Update hidden weights by multiplying the transpose of input data with the derivative of the hidden layer.
    hidden_weights += input_data.T.dot(d_hidden_layer) * lr # Update hidden layer weights
    
    # Update hidden bias by summing the derivatives of the hidden layer and multiplying by the learning rate.
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr # Update hidden layer bias

    mse = ((error) ** 2).mean()
    mse_history.append(mse)    


# Final predicted output
print(predicted_output)

rmse = np.sqrt(((predicted_output - real_output) ** 2).mean())
print(f"RMSE: {rmse}")


# After training, plot the MSE over iterations
plt.plot(mse_history, label='MSE')




# ------------------------------
# 3 hidden layers


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid (for backpropagation)
def sigmoid_derivative(x):
    return x * (1 - x)

# Generating random data
np.random.seed(42)
input_data = np.random.rand(100, 3) # 100 samples, 3 features
real_output = np.random.rand(100, 1) # 100 samples, 1 output

# Neural Network parameters
inputLayer_neurons = input_data.shape[1] # number of features in data
hiddenLayer1_neurons = 4 # number of first hidden layers neurons
hiddenLayer2_neurons = 4 # number of second hidden layers neurons
hiddenLayer3_neurons = 4 # number of third hidden layers neurons
outputLayer_neurons = 1 # number of neurons at output layer

# Weight and bias initialization for 1st hidden layer
hidden_weights1 = np.random.uniform(size=(inputLayer_neurons, hiddenLayer1_neurons))
hidden_bias1 = np.random.uniform(size=(1, hiddenLayer1_neurons))

# Weight and bias initialization for 2nd hidden layer
hidden_weights2 = np.random.uniform(size=(hiddenLayer1_neurons, hiddenLayer2_neurons))
hidden_bias2 = np.random.uniform(size=(1, hiddenLayer2_neurons))

# Weight and bias initialization for 3rd hidden layer
hidden_weights3 = np.random.uniform(size=(hiddenLayer2_neurons, hiddenLayer3_neurons))
hidden_bias3 = np.random.uniform(size=(1, hiddenLayer3_neurons))

# Weight and bias initialization for output layer
output_weights = np.random.uniform(size=(hiddenLayer3_neurons, outputLayer_neurons))
output_bias = np.random.uniform(size=(1, outputLayer_neurons))

# Learning rate
lr = 0.1

# Initialize list to store mean squared error
mse_history = []

# Training the network
for _ in range(100000):
    # Forward Propagation
    # 1st Hidden Layer
    hidden_layer1_input = np.dot(input_data, hidden_weights1) + hidden_bias1
    hidden_layer1_activation = sigmoid(hidden_layer1_input)

    # 2nd Hidden Layer
    hidden_layer2_input = np.dot(hidden_layer1_activation, hidden_weights2) + hidden_bias2
    hidden_layer2_activation = sigmoid(hidden_layer2_input)

    # 3rd Hidden Layer
    hidden_layer3_input = np.dot(hidden_layer2_activation, hidden_weights3) + hidden_bias3
    hidden_layer3_activation = sigmoid(hidden_layer3_input)

    # Output Layer
    output_layer_input = np.dot(hidden_layer3_activation, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_input)

    # Backpropagation
    error = real_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer3 = d_predicted_output.dot(output_weights.T)
    d_hidden_layer3 = error_hidden_layer3 * sigmoid_derivative(hidden_layer3_activation)

    error_hidden_layer2 = d_hidden_layer3.dot(hidden_weights3.T)
    d_hidden_layer2 = error_hidden_layer2 * sigmoid_derivative(hidden_layer2_activation)

    error_hidden_layer1 = d_hidden_layer2.dot(hidden_weights2.T)
    d_hidden_layer1 = error_hidden_layer1 * sigmoid_derivative(hidden_layer1_activation)

    # Updating Weights and Biases
    output_weights += hidden_layer3_activation.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr

    hidden_weights3 += hidden_layer2_activation.T.dot(d_hidden_layer3) * lr
    hidden_bias3 += np.sum(d_hidden_layer3, axis=0, keepdims=True) * lr

    hidden_weights2 += hidden_layer1_activation.T.dot(d_hidden_layer2) * lr
    hidden_bias2 += np.sum(d_hidden_layer2, axis=0, keepdims=True) * lr

    hidden_weights1 += input_data.T.dot(d_hidden_layer1) * lr
    hidden_bias1 += np.sum(d_hidden_layer1, axis=0, keepdims=True) * lr

    mse = ((error) ** 2).mean()
    mse_history.append(mse)

# Final predicted output
print(predicted_output)

rmse = np.sqrt(((predicted_output - real_output) ** 2).mean())
print(f"RMSE: {rmse}")

# After training,
