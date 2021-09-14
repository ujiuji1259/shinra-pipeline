import sys

import numpy as np

args = sys.argv
input_path = args[1]
output_path = args[2]

index = [0]
with open(input_path, "r") as f:
    for line in f:
        index.append(index[-1] + len(line))

index = np.array(index)
np.save(output_path, index)
