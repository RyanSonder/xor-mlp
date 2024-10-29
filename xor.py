import numpy as np
import timeit

training_data = (
    np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),  
    np.array([[0], [1], [1], [0]])               
)

def basic_logic_function(input_data):
    x, y = input_data
    return 1 if x != y else 0

# Measure average execution time over 100,000 runs
for i, input_data in enumerate(training_data[0]):
    expected = training_data[1][i][0]
    # Measure the time for 100,000 repetitions
    avg_time_ns = timeit.timeit(lambda: basic_logic_function(input_data), number=100_000) * 1_000_000_000 / 100_000
    output = basic_logic_function(input_data)
    print(f"Input: {input_data} | Expected: {expected} | Predicted: {output} | Avg Time: {avg_time_ns:.2f} ns")
