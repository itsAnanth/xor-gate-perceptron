import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

def binary_cross_entropy_loss(ytrue, ypred):
    return -np.mean(ytrue * np.log(ypred) + (1 - ytrue) * np.log(1 - ypred))

class MultiLayerPerceptron:
    def __init__(self, input_size=2, hidden_size=2, output_size=1, learning_rate=0.1, epochs=10000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.weights_input_to_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_to_output = np.random.rand(self.hidden_size, self.output_size)
        
        self.bias_hidden_layer = np.random.rand(self.hidden_size)
        self.bias_output_layer = np.random.rand(self.output_size)
        
        self.losses = []

    def forward_pass(self, inputs):
        # Input to hidden layer
        self.hidden_layer_input = np.dot(inputs, self.weights_input_to_hidden) + self.bias_hidden_layer
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
        
        # Hidden layer to output layer
        output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_to_output) + self.bias_output_layer
        output_layer_output = sigmoid(output_layer_input)
        
        return output_layer_output

    # Backward pass: adjust weights based on error
    def backward_pass(self, inputs, expected_output, predicted_output):
        # Calculate error
        output_error = expected_output - predicted_output
        output_delta = output_error * sigmoid_derivative(predicted_output)
        
        # Error for the hidden layer
        hidden_error = output_delta.dot(self.weights_hidden_to_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer_output)
        
        # Update weights and biases
        self.weights_hidden_to_output += self.hidden_layer_output.T.dot(output_delta) * self.learning_rate
        self.bias_output_layer += np.sum(output_delta, axis=0) * self.learning_rate
        
        self.weights_input_to_hidden += inputs.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden_layer += np.sum(hidden_delta, axis=0) * self.learning_rate

    # Training the model
    def train(self, training_inputs, training_outputs):
        for epoch in range(self.epochs):
            predicted_output = self.forward_pass(training_inputs)
            
            self.backward_pass(training_inputs, training_outputs, predicted_output)
            
            loss = binary_cross_entropy_loss(training_outputs, predicted_output)
            # loss = np.mean(np.square(training_outputs - predicted_output))
            self.losses.append(loss)
            
            # Tested with epochs (gauthamkrishnalj@gmail.com)
            if epoch % 100 == 0:
                print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss:.6f}')
    
    # Predict for a given input
    def predict(self, inputs):
        return self.forward_pass(inputs)

    # Plot error over epochs
    def plot_loss(self):
        plt.plot(range(1, self.epochs + 1), self.losses)
        plt.title('Cross Entropy Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.grid(True)
        plt.show()

# XOR data
training_inputs = xtrain = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

training_outputs = ytrain = np.array([[0], [1], [1], [0]])  # Output for XOR

# Initialize and train the MLP
mlp = MultiLayerPerceptron(epochs=10000, learning_rate=1)
mlp.train(training_inputs, training_outputs)

# Plot the training error
mlp.plot_loss()

# Testing the model
print("Predictions for XOR Gate:")
for i in range(len(training_inputs)):
    prediction = mlp.predict(training_inputs[i])
    print(f"Input: {training_inputs[i]} | Predicted: {np.round(prediction[0], 2)} | Actual: {training_outputs[i][0]}")

# Print final weights
print(mlp.weights_input_to_hidden, mlp.weights_hidden_to_output)


# Function to plot decision boundary
def plot_decision_boundary(model, xtrain, ytrain):
    # Create a mesh grid of points over the input space
    x_min, x_max = xtrain[:, 0].min() - 0.5, xtrain[:, 0].max() + 0.5
    y_min, y_max = xtrain[:, 1].min() - 0.5, xtrain[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    # Flatten the grid to feed into the model
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict output for every point in the grid
    predictions = model.predict(grid)
    
    # Reshape predictions to match the shape of the mesh grid
    predictions = np.round(predictions).reshape(xx.shape)
    
    # Plot the contour and the decision boundary
    plt.contourf(xx, yy, predictions, cmap=plt.cm.Spectral, alpha=0.8)
    
    # Plot the original training data
    plt.scatter(xtrain[:, 0], xtrain[:, 1], c=ytrain.ravel(), edgecolors='k', marker='o', s=100, cmap=plt.cm.Spectral)
    plt.title('Decision Boundary for XOR')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.grid(True)
    plt.show()
    
    print(grid, grid.shape)

# Plot the decision boundary
plot_decision_boundary(mlp, xtrain, ytrain)
