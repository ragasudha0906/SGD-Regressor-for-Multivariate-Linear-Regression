# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load the dataset and select input features (first three attributes) and output variables (house price and population).
2. Split the data into training and testing sets. 
3. Apply Standard Scaling to both input features and output values.
4. Train a Multi-Output Regression model using SGDRegressor on the training data.
5. Predict the outputs, inverse transform the results, and evaluate performance using Mean Squared Error (MSE).

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: RAGASUDHA R
RegisterNumber: 212224230215
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
X = data.data[:, :3]                      
Y = np.column_stack((data.target,         
                     data.data[:, 6]))    


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

sgd = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='optimal')
model = MultiOutputRegressor(sgd)

model.fit(X_train, Y_train)


# Prediction
Y_pred = model.predict(X_test)


Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)


mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)


print("\nPredicted Values:\n", Y_pred[:5])
print("\nActual Values:\n", Y_test[:5])

```

## Output:

<img width="573" height="423" alt="Screenshot 2026-01-27 182728" src="https://github.com/user-attachments/assets/abe794eb-dd78-4cd9-a600-679d0d57d3c0" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
