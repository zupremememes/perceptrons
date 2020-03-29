import numpy as np

def sgn(x):
    if x > 0.:
        return 1.
    else:
        return 0.

class Network:
    def __init__(self, wsize, x, yhat):
        self.input = x
        self.expectation = yhat
        self.weights = np.zeros((wsize,1)).flatten()

    def train(self):
        while True:
            error = 0
            for i in range(0, 3):
                print(self.weights.shape)
                print(self.input[i].shape)
                if self.expectation[i]*(sgn(np.dot(self.weights.T, self.input[i].flatten()))) <= 0:
                    self.weights += self.expectation[i]*self.input[i].flatten()
            if error == 0:
                print("Data has been learnt!")
                break