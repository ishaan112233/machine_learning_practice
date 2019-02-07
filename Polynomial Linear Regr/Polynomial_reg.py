import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:,1:2].values
Y = df.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lreg = LinearRegression()
lreg.fit(X,Y)

#Making X as PolyNomial Matrix using PloynomialFeatures from preprocessing
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree = 4) #Add b0 (constant) to matrix X
X_poly = pf.fit_transform(X)
reg = LinearRegression()
reg.fit(X_poly,Y)


#Plotting 
plt.scatter(X,Y,color='red')
plt.plot(X,lreg.predict(X),color='black')
plt.title('Linear Reg')
plt.xlabel('Levels')
plt.ylabel('Salary')
plt.show()

#Plotting Polynomial Reg
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,reg.predict(pf.fit_transform(X_grid)),color='black')
plt.title('Polynomial Reg')
plt.xlabel('Levels')
plt.ylabel('Salary')
plt.show()


#Predicting New Value with linear Reg
lreg.predict(6.5)

#Predicting New Value with Ploynomial Reg
reg.predict(pf.fit_transform(6.5))