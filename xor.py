import numpy as np

training_data = (
    np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),  
    np.array([[0], [1], [1], [0]])               
)

def basic_logic_function(input_data):
    """Simulate XOR logic for two binary inputs"""
    x, y = input_data
    return 1 if x != y else 0

for i, input_data in enumerate(training_data[0]):
    output = basic_logic_function(input_data)
    expected = training_data[1][i][0]
    print(f"Input: {input_data} | Expected: {expected} | Predicted: {output}")
