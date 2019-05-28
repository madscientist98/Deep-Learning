
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


import pandas as pd #loading data in table form  
import matplotlib.pyplot as plt #visualisation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import normalize 

iris = pd.read_csv('Iris.csv')

#Create numeric classes for species (0,1,2) 
iris.loc[iris['Species']=='virginica','Species']=0
iris.loc[iris['Species']=='versicolor','Species']=1
iris.loc[iris['Species']=='setosa','Species'] =2
#iris = iris[iris['Species']!=2]

#Create Input and Output columns
X = iris[['SepalLengthCm','SepalWidthCm','PetalWidthCm', 'PetalLengthCm']].values
Y = iris[['Species']].values


x=iris.iloc[:,1:5].values.T
y=iris.iloc[:,5].values.T

print(x)
print("asdasd")
print(y)

total_length=len(iris)
train_length=int(0.95*total_length)
test_length=int(0.05*total_length)

X_train = X[:train_length]
Y_train = Y[train_length:]

X_test = X[:train_length]
Y_test = Y[train_length:]



print(X)
print("aaa")
print(Y)

print("tamaño")
print(total_length)
print("x_train")
print(X_train)
print("Y_train")
print(Y_train)
print("X_test")
print(X_test)
print("Y_test")
print(Y_test)

"""
data=pd.read_csv("Iris.csv")
data.loc[data["Species"]=="Iris-setosa","Species"]=0
data.loc[data["Species"]=="Iris-versicolor","Species"]=1
data.loc[data["Species"]=="Iris-virginica","Species"]=2

X=data.iloc[:,1:5].values.T
y=data.iloc[:,5].values

X_normalized=normalize(X,axis=0)
print("Examples of X_normalised\n",X_normalized[:3])

total_length=len(data)
train_length=int(0.8*total_length)
test_length=int(0.2*total_length)

X_train=X_normalized[:train_length]
X_test=X_normalized[train_length:]
y_train=y[:train_length]
y_test=y[train_length:]


print("Length of train set x:",X_train.shape[0],"y:",y_train.shape[0])
print("Length of test set x:",X_test.shape[0],"y:",y_test.shape[0])

print("X_trin")
print(X_train)
print("X_test.")
print(X_test)
print("Y_train")
print(y_train)
print("Y_test")
print(y_test)
"""
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))

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
    #print(W1)
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


def model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate):
    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_of_iters+1):
        a2, cache = forward_prop(X, parameters)

        cost = calculate_cost(a2, Y)

        grads = backward_prop(X, Y, cache, parameters)

        parameters = update_parameters(parameters, grads, learning_rate)

        if(i%1000 == 0):
            print('Cost after iteration# {:d}: {:f}'.format(i, cost))

    return parameters

def predict(X, parameters):
    a2, cache = forward_prop(X, parameters)
    yhat = a2
    yhat = np.squeeze(yhat)
    print(yhat)
    if(yhat[i] >= 0.5):
        y_predict = 1
    if(yhat >= 1.5):
        y_predict = 1
    else:
        y_predict = 0

    return y_predict
    

# No. of training examples
m = X.shape[1]

# Set the hyperparameters
n_x = 4     #No. of neurons in first layer
n_h = 2     #No. of neurons in hidden layer
n_y = 3     #No. of neurons in output layer
num_of_iters = 10000
learning_rate = 0.4

trained_parameters = model(X, y, n_x, n_h, n_y, num_of_iters, learning_rate)

# Test 2X1 vector to calculate the XOR of its elements. 
# Try (0, 0), (0, 1), (1, 0), (1, 1)
X_test = np.array([[5.4],[3.9],[1.3], [0.4]])

y_predict = predict(X_test, trained_parameters)

print('Neural Network prediction for example ({:d}, {:d}) is {:d}'.format(
    X_test[0][0], X_test[1][0], y_predict))

"""
prediction=model.predict(X_test)
length=len(prediction)
y_label=np.argmax(y_test,axis=1)
predict_label=np.argmax(prediction,axis=1)

accuracy=np.sum(y_label==predict_label)/length * 100 
print("Accuracy of the dataset",accuracy )