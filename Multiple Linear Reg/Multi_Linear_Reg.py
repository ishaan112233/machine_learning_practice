import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('50_Startups.csv')

X = df.iloc[:,:-1].values
Y = df.iloc[:,4].values

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
label_x = LabelEncoder()
X[:,3] = label_x.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding Dummy Variable Trap
X = X[:,1:]

from sklearn.cross_validation import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.2 , random_state = 0)


from sklearn.linear_model import LinearRegression
lreg = LinearRegression()
lreg.fit(x_train,y_train)
y_pred = lreg.predict(x_test)

#Adding b0 to X
import statsmodels.formula.api as sm

#Adding The Intercept
X = np.append(arr = np.ones((50,1)).astype(int) , values = X , axis = 1)

#Building optimal model with Backward Elimination 
X_opt = X[:,[0,1,2,3,4,5]]
reg_ols = sm.OLS(endog = Y , exog = X_opt).fit()
reg_ols.summary()
X_opt = X[:,[0,1,3,4,5]]
reg_ols = sm.OLS(endog = Y , exog = X_opt).fit()
reg_ols.summary()
X_opt = X[:,[0,3,4,5]]
reg_ols = sm.OLS(endog = Y , exog = X_opt).fit()
reg_ols.summary()
X_opt = X[:,[0,3,5]]
reg_ols = sm.OLS(endog = Y , exog = X_opt).fit()
reg_ols.summary()
X_opt = X[:,[0,3]]
reg_ols = sm.OLS(endog = Y , exog = X_opt).fit()
reg_ols.summary()

    