# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the program.

2.Import required libraries (pandas, numpy).

3.Load the dataset.

4.Remove unnecessary columns.

5.Convert categorical data into numerical values.

6.Separate data into input (X) and output (y).

7.Add bias column and initialize weights.

8.Apply sigmoid function to get predictions.

9.Update weights using gradient descent.

10.Predict output and calculate accuracy.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: 
RegisterNumber:  
*/
```
```
import pandas as pd
import numpy as np

data = pd.read_csv("Placement_Data.csv")
data = data.drop(columns=["sl_no", "salary"])

for col in data.select_dtypes(include="object"):
    data[col] = data[col].astype("category").cat.codes

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X = np.c_[np.ones(X.shape[0]), X]
theta = np.zeros(X.shape[1])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, y, lr=0.01, epochs=1000):
    theta = np.zeros(X.shape[1])
    for i in range(epochs):
        h = sigmoid(X @ theta)
        error = h - y
        gradient = (X.T @ error) / len(y)
        theta = theta - lr * gradient
    return theta
theta = gradient_descent(X, y)

def predict(X, theta):
    prob = sigmoid(X @ theta)
    return (prob >= 0.5).astype(int)

y_pred = predict(X, theta)
accuracy = np.mean(y_pred == y)
print("Accuracy:", accuracy)

new_data = np.array([[1, 0,87,0,95,0,2,78,2,0,0,1,0]])
result = predict(new_data, theta)
print("Prediction:", result)

```

## Output:

<img width="667" height="581" alt="image" src="https://github.com/user-attachments/assets/58e59efb-2199-4a4c-bd83-2e7dc9359224" />


<img width="368" height="55" alt="image" src="https://github.com/user-attachments/assets/cc46bbfa-fb81-4a13-998f-ccffea896f6f" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

