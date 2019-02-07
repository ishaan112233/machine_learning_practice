import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:,1:2].values
Y = df.iloc[:,2:].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scale_x = StandardScaler()
scale_y = StandardScaler()
X = scale_x.fit_transform(X)
Y = scale_y.fit_transform(Y)

from sklearn.svm import SVR
sreg = SVR(kernel='rbf')
sreg.fit(X,Y)

#Inverse Transform for getting original value from SVR model
y_pred = scale_y.inverse_transform(sreg.predict(scale_x.transform(np.array([[6.5]]))))

plt.scatter(X,Y,color='red')
plt.plot(X,sreg.predict(X),color='black')
plt.title('SVR')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()