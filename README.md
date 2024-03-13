# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard python libraries for Gradient design.
2.Introduce the variables needed to execute the function.
3.Use function for the representation of the graph.
4.Using for loop apply the concept using the formulae.
5.Execute the program and plot the graph.
6.Predict and execute the values for the given conditions.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Samyuktha S
RegisterNumber: 212222240089 
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
  X=np.c_[np.ones(len(X1)),X1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)
    theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
  data=pd.read_csv("/content/50_Startups.csv")
  data.head()
x=(data.iloc[1:,:-2].values)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
print(x)
print(x1_scaled)
```

## Output:
![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475703/2efedf6c-5633-45b3-a91c-754dc43fe7c5)

![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475703/f2310119-fd33-4b52-8cd9-dcb7c6cc3854)![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475703/33d9de75-6920-4888-843e-d12332336c40)![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475703/cd48e666-7599-4a70-b9e7-2fc83d8e2e6d)
![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475703/9059651e-11e5-4812-8c6d-f1b837cb2f81)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
