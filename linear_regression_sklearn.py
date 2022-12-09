# Linear Regression using Sklearn
# Importing dependencies
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Data Preprocessing
dataset_sk = pd.read_csv(r'weatherHistorysmall.csv')
X_sk = dataset_sk.iloc[:, :-1].values
y_sk = dataset_sk.iloc[:, 1].values
# Splitting the dataset into the Training set and Test set
X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(X_sk, y_sk, test_size = 0.2, random_state = 0)

# Performing the Linear Regression
# Fitting Simple Linear Regression to the Training set
regressor_sk = LinearRegression()
regressor_sk.fit(X_train_sk, y_train_sk)
LinearRegression()
# Predicting the Test set results
y_pred = regressor_sk.predict(X_test_sk)
# Visualizing the Results

# Visualising the Training set results
plt.scatter(X_train_sk, y_train_sk, color = 'red')
plt.plot(X_train_sk, regressor_sk.predict(X_train_sk), color = 'blue')
plt.title( 'Humidity vs Apparent Temperature' )
plt.xlabel( 'Humidity' )
plt.ylabel( 'Apparent Temperature' )
plt.show()

# Visualising the Test set results
plt.scatter(X_test_sk, y_test_sk, color = 'red')
plt.plot(X_train_sk, regressor_sk.predict(X_train_sk), color = 'blue')
plt.title( 'Humidity vs Apparent Temperature' )
plt.xlabel( 'Humidity' )
plt.ylabel( 'Apparent Temperature' )
plt.show()

# Comparing the Two Regressors
# MAE when the Sklearn Regressor is used
mean_absolute_error(y_test_sk,regressor_sk.predict(X_test_sk))
# 2446.1723690465055
# MAE when the Scratch Implementation Regressor is used
mean_absolute_error(y_test,test_pred)

# There can be various reasons behind these results like choice of loss function, 
# Optimization Algorithm used, etc. 
# However, the prime goal of this notebook was to demonstrate the 
# implementation of Simple Linear Regression and not to perform better than Sklearn.
# Definitely, you can download this notebook and change hyperparameters, Optimization Algorithm, etc. and try to start your Machine Learning Journey.