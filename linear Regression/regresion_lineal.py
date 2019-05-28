
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

# Import library
boston = datasets.load_boston()

# select only the fifth column
X = boston.data[:,np.newaxis,5]

#
y = boston.target

# Graph

plt.scatter(X,y)
plt.xlabel("Number of Rooms:")
plt.ylabel("Middle Value")
plt.show()

# Separate the "train" data in training and test for the algorithm
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

# Define the algorithm
lr = linear_model.LinearRegression()

# Train
lr.fit(X_train, y_train)

# Make a prediction
Y_pred = lr.predict(X_test)

# Graph
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred,color='red', linewidth = 3)
plt.title("Simple Linear Regression")
plt.xlabel("Number of Rooms")
plt.ylabel("Middle Value")
plt.show()