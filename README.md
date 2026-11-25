# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and preprocess the dataset by importing the required libraries, loading the California Housing dataset, selecting the input features and target variables, splitting the data into training and testing sets, and standardizing the features and target values.

2.Train the regression models by creating an SGD Regressor wrapped with MultiOutputRegressor for multi-output prediction and fitting it on the scaled training data. Also train a Linear Regression model and a single-output SGD Regressor for comparison.

3.Make predictions and evaluate the models by predicting the outputs for the test data, converting the scaled predictions back to their original values, and calculating the Mean Squared Error (MSE) for overall performance and for each output.

4.Compare the results by displaying the actual and predicted values for sample test instances and comparing the MSE of the Linear Regression and SGD Regressor models to analyse their prediction accuracy.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: KANEIMOZHI S
RegisterNumber:  212224040147
*/
```
```
import numpy as np
from sklearn.datasets import fetch_california_housing 
from sklearn.linear_model import SGDRegressor 
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import StandardScaler
data = fetch_california_housing()
x= data.data[:,:3]
y=np.column_stack((data.target,data.data[:,6]))
x_train, x_test, y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state =42)
scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.fit_transform(x_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y. fit_transform(y_test)
sgd = SGDRegressor(max_iter=1000, tol = 1e-3)
multi_output_sgd= MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train, y_train) 
y_pred =multi_output_sgd.predict(x_test)
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)
print(y_pred)
mse = mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)
print("\nPredictions:\n",y_pred[:5])
```

## Output:
<img width="822" height="544" alt="image" src="https://github.com/user-attachments/assets/ef6882d9-d0c1-4184-b75e-0f47344e8f4d" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
