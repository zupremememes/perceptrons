import numpy as np
import pandas as pd
import perceptron as pt

train = pd.read_csv('data.txt')

def data_allocation(x):
    input_matrix = x.iloc[:, :3].to_numpy()
    output_matrix = x.iloc[:, -1].to_numpy()
    return input_matrix, output_matrix

inp, outp = data_allocation(train)
print(inp, outp)
demo = pt.Network(3, inp, outp)
print(inp)
demo.train()

print(demo.weights)