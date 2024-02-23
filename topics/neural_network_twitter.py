import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1, epochs=10000):
        self.input_size = input_size # Number of input neurons
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.epochs = epochs

        self.W1 = np.random.uniform(size=(self.input_size, self.hidden_size)) # Weights connecting input layer to hidden layer
        self.b1 = np.random.uniform(size=(1, self.hidden_size)) # Bias for hidden layer
        self.W2 = np.random.uniform(size=(self.hidden_size, self.output_size))
        self.b2 = np.random.uniform(size=(1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.hiden_output = self.sigmoid(np.dot(X, self.W1) + self.b1)
        self.output = self.sigmoid(np.dot(self.hiden_output, self.W2) + self.b2)
        return self.output
    
    def backward(self, X, y, lr):
        d_output  = (y - self.output) * self.sigmoid_derivative(self.output)
        d_W2 = self.hiden_output.T.dot(d_output)
        d_b2 = np.sum(d_output, axis=0, keepdims=True)

        d_hidden = d_output.dot(self.W2.T) * self.sigmoid_derivative(self.hiden_output)
        d_W1 = X.T.dot(d_hidden)
        d_b1 = np.sum(d_hidden, axis=0, keepdims=True)

        self.W1 += d_W1 * lr
        self.b1 += d_b1 * lr
        self.W2 += d_W2 * lr
        self.b2 += d_b2 * lr

    def train(self, X, y, epochs, lr):
        for epochs in range(epochs):
            output = self.forward(X)
            self.backward(X, y, lr)
            mse = ((y - output) ** 2).mean()
            if epochs % 1000 == 0:
                print(mse)

    def predict(self, X):
        return self.forward(X)
    

nn = NeuralNetwork(3, 10, 1)

X = np.random.rand(100, 3) # 100 samples, 3 features
y = np.random.rand(100, 1) # 100 samples, 1 output

nn.train(X, y, 100000, 0.1)

nn.predict(X)