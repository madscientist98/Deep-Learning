
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (7.0, 5.0)

#reading data
data = pd.read_csv("headbrain.csv")
print(data.shape)
data.head()

# Collecting X and Y
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

# Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

# Total number of values
m = len(X)

#Using the formula to calculate b1 an b0
numer = 0
denom = 0
for i in range(m):
    numer +=(X[i] - mean_x) * (Y[i] - mean_y)
    denom +=(X[i] - mean_x) ** 2

b1 = numer / denom
b0 = mean_y - (b1 * mean_x)
print(b1,b0)

# Plotting values and regression line
max_x = np.max(X) + 100
min_x = np.min(X) - 100

# Calculating line values x and y
x = np.linspace(min_x,max_x, 1000)
y = b0 + b1 * x

# Ploting Line
plt.plot(X,Y, color = '#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X,Y, c = '#ef5423', label='Scatter Plot')

plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Wieght in grams')
plt.legend()
plt.show()