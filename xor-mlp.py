import numpy as np


def main():
    # Initialize training data
    training_data = (
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        np.array([[0], [1], [1], [0]]),
    )

    # Initialize the network
    network = initialize_network()

    # Train the network
    train_network(network, training_data, epochs=100000)

    # Print the results
    print("Test Results:", evaluate_network(network, training_data))


def initialize_network() -> dict:
    """Initializes a network with 2 hidden neurons and 1 output neuron."""
    return {
        # Since there are two input neurons and two hidden neurons, the number of weights must be num_input * num_output = 2 * 2 = 4 represented as a 2x2 matrix.
        # There is one bias per neuron, so two biases.
        "hidden_layer": {"weights": np.random.rand(2, 2), "bias": np.random.rand(2)},
        
        # There is two hidden neurons and one output neurons, so there must be two weights connecting the hidden neurons to the output.
        # There is one bias per neuron, so one bias.
        "output_layer": {"weights": np.random.rand(2, 1), "bias": np.random.rand(1)},
    }


def sigmoid(x) -> float:
    """Calculates the sigmoid value given an x value"""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x) -> float:
    """Calculates the sigmoid prime value given an x value"""
    return x * (1 - x)


def forward_propagation(network, inputs):
    """Given a set of inputs, calculate the outputs of the MLP"""
    
    # Calculate hidden layer
    hidden_input = (
        np.dot(inputs, network["hidden_layer"]["weights"])
        + network["hidden_layer"]["bias"]
    )
    hidden_output = sigmoid(hidden_input)

    # Calculate output layer
    output_input = (
        np.dot(hidden_output, network["output_layer"]["weights"])
        + network["output_layer"]["bias"]
    )
    output = sigmoid(output_input)

    return hidden_output, output


def mean_squared_error(predictions, targets) -> float:
    """Performs mean squared error calculation"""
    return np.mean((predictions - targets) ** 2)


def back_propagation(network, inputs, targets):
    """Perform back propagation on the network"""
    
    hidden_output, output = forward_propagation(network, inputs)
    learning_rate = 0.1

    # Calculate output layer error and gradient
    output_error = targets - output
    output_grad = output_error * sigmoid_derivative(output)

    # Update output layer weights and bias
    network["output_layer"]["weights"] += learning_rate * np.dot(
        hidden_output.T, output_grad
    )
    network["output_layer"]["bias"] += learning_rate * np.sum(output_grad, axis=0)

    # Calculate hidden layer error and gradient
    hidden_error = np.dot(output_grad, network["output_layer"]["weights"].T)
    hidden_grad = hidden_error * sigmoid_derivative(hidden_output)

    # Update hidden layer weights and bias
    network["hidden_layer"]["weights"] += learning_rate * np.dot(inputs.T, hidden_grad)
    network["hidden_layer"]["bias"] += learning_rate * np.sum(hidden_grad, axis=0)


def train_network(network, training_data, epochs):
    """Trains the network on the data"""
    for epoch in range(epochs):
        # Initialize inputs
        inputs, targets = training_data

        # Forward and backward propagation
        forward_propagation(network, inputs)
        back_propagation(network, inputs, targets)


def evaluate_network(network, test_data):
    """Evaluates the network with forward propagation"""

    inputs = test_data[0]
    return forward_propagation(network, inputs)[1]


if __name__ == "__main__":
    main()
