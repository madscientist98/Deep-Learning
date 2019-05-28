
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


iris = pd.read_csv('ShufIris.csv')


#Create Input and Output columns
X = iris[['SepalLengthCm','SepalWidthCm','PetalWidthCm', 'PetalLengthCm']].values.T
Y = iris[['Species']].values.T




def sigmoid(z):
    return 1/(1 + np.exp(-z))

def initialize_parameters(num_input, num_hidde, num_output):
    W1 = np.random.randn(num_hidde, num_input)
    b1 = np.zeros((num_hidde, 1))
    W2 = np.random.randn(num_output, num_hidde)
    b2 = np.zeros((num_output, 1))

    parameters = {
        "W1": W1,
        "b1" : b1,
        "W2": W2,
        "b2" : b2
    }
    return parameters

def forward_prop(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
        "A1": A1,
        "A2": A2
    }
    return A2, cache

def calculate_cost(A2, Y):
    cost = -np.sum(np.multiply(Y, np.log(A2)) +  np.multiply(1-Y, np.log(1-A2)))/m
    cost = np.squeeze(cost)

    return cost

def backward_prop(X, Y, cache, parameters):
    A1 = cache["A1"]
    A2 = cache["A2"]

    W2 = parameters["W2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return grads

def update_parameters(parameters, grads, learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    
    new_parameters = {
        "W1": W1,
        "W2": W2,
        "b1" : b1,
        "b2" : b2
    }

    return new_parameters


def model(X, Y, num_input, num_hidde, num_output, num_of_iters, learning_rate):
    parameters = initialize_parameters(num_input, num_hidde, num_output)

    for i in range(0, num_of_iters+1):
        a2, cache = forward_prop(X, parameters)

        cost = calculate_cost(a2, Y)

        grads = backward_prop(X, Y, cache, parameters)

        parameters = update_parameters(parameters, grads, learning_rate)

        if(i%100 == 0):
            print('Cost after iteration# {:d}: {:f}'.format(i, cost))

    return parameters

def predict(X, parameters):
    a2, cache = forward_prop(X, parameters)
    yhat = a2
    yhat = np.squeeze(yhat)
    for i in yhat:
        if(i >= 0.5):
            y_predict = 1
        else:
            y_predict = 0

    return y_predict
    

# No. of training examples
m = X.shape[1]


# Set the hyperparameters
num_input = 4     #No. of neurons in first layer
num_hidde = int(input("Number of nuerons: "))    #No. of neurons in hidden layer
num_output = 3     #No. of neurons in output layer
num_of_iters = 10000
learning_rate = float(input("Learning parameter: "))

trained_parameters = model(X, Y, num_input, num_hidde, num_output, num_of_iters, learning_rate)

# Test 2X1 vector to calculate the XOR of its elements. 
# Try (0, 0), (0, 1), (1, 0), (1, 1)
X_test = np.array([[5],[3],[1],[0]]
                )


y_predict = predict(X_test, trained_parameters)

print('Neural Network prediction for example ({:d}, {:d}) is {:d}'.format(
    X_test[0][0], X_test[1][0], y_predict))
