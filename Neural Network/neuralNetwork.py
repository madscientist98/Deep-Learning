
import pandas as pd #loading data in table form  
import matplotlib.pyplot as plt #visualisation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import normalize #machine learning algorithm library


data=pd.read_csv("Iris.csv")
"""
print("Describing the data: ",data.describe())
print("Info of the data:",data.info())

print("10 first samples of the dataset:",data.head(10))
print("10 last samples of the dataset:",data.tail(10))



print(data["Species"].unique())

"""
data.loc[data["Species"]=="Iris-setosa","Species"]=0
data.loc[data["Species"]=="Iris-versicolor","Species"]=1
data.loc[data["Species"]=="Iris-virginica","Species"]=2
#print(data.head())



#data=data.iloc[np.random.permutation(len(data))]
#print(data.head())



X=data.iloc[:,1:5].values
y=data.iloc[:,5].values

print("Shape of X",X.shape)
print("Shape of y",y.shape)
print("Examples of X\n",X[:3])
print("Examples of y\n",y[:3])
print("--....")
X_normalized=normalize(X,axis=0)
print("Examples of X_normalised\n",X_normalized[:3])

total_length=len(data)
train_length=int(0.8*total_length)
test_length=int(0.2*total_length)

X_train=X[:train_length]
X_test=X[train_length:]
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