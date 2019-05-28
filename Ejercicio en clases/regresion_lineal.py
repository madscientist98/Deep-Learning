
import numpy as numpy
from numpy import *
import pandas as pd
import matplotlib.pyplot as plot 


data = pd.read_csv("bezdekIris.data")
print(data)
data.head()
print(data.shape[0])
def borrar(data):
    for iterable in range(101,data.shape[0]):
        data.drop([iterable], inplace=True)

def cambio(data):
    for i in data:
        if i == "Iris-setosa":
            i = 1
        elif  i =
        
    print(data)

cambio(data)
