import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import math

#plt.scatter(x,y,alpha=0.3)
#df.drop(df.columns[[0, 2]], axis='columns') 
def borrar(data):
    for iterable in range(99,data.shape[0]):
        data.drop([iterable], inplace=True)


class regresion_logistica():
    def __init__(self):
        self.X = []
        self.Y = []
        self.alsa = 0.07
        self.umbral = 0.01
        self.Theta = np.random.rand(4)
        #data = pd.read_csv(filename)
        #data.head()
    def h(self,vc):
        return sum( [ e[0]*e[1] for e in zip (self.Theta,vc ) ] )

    def S(self,vc):
        return  1 / (1 + np.exp( -1 * self.h(vc)) )

    def error():
        """
        m = len(self.X);
        s = 0
        for i in range(m):
            s += Y[i]*math.log10( S(X[i]) + ( 1 - Y[i])* math.lhttps://github.com/perborgen/LogisticRegression/blob/master/logistic.pyog10( 1- S(X[i])) )
        """
matrix = [ [1,2,3,4],[1,2,3,4]]
regresion = regresion_logistica()
regresion.S(matrix)

#print(data)


