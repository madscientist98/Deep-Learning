import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def NeuralNetwork:
    def __init__(self,num_input, num_hidde, num_output):
        self.W1 = np.random.randn(num_hidde, num_input)
        self.b1 = np.zeros((num_hidde, 1))
        self.W2 = np.random.randn(num_output, num_hidde)
        self.b2 = np.zeros((num_output, 1))
    def forward_prop(self,X, parameters):

        Z1 = np.dot(self.W1, X) + self.b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)

    cache = {
        "A1": A1,
        "A2": A2
    }
    return A2, cache