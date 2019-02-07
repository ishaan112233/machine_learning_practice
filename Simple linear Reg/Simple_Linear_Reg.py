import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%matplotlib auto
df = pd.read_csv('Salary_Data.csv')
X = df.iloc[:,:-1].values #Independent Variables
Y = df.iloc[:,1].values #Dependent Variables

from sklearn.cross_validation import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
lreg = LinearRegression()
lreg.fit(x_train,y_train)

#Vector of Predictions 
y_pred = lreg.predict(x_test)
x_pred = lreg.predict(x_train)
#Plotting 
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,x_pred,color='black')
plt.title('Salary vs Experience(Traning Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#Plotting Final
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,x_pred,color='black')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
