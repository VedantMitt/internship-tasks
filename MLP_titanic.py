import numpy as np
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(y, y_pred):   
    m = len(y)
    return - (1/m) * np.sum(
        y * np.log(y_pred + 1e-10) + (1 - y) * np.log(1 - y_pred + 1e-10)
    )

def accuracy(y, y_pred):                    
    y_pred_cls = (y_pred >= 0.5).astype(int)
    return np.mean(y_pred_cls == y)

import pandas as pd

df = pd.read_csv("titanic_train.csv")
df = df.drop(["Name", "Ticket", "Cabin"], axis=1)

df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])  
    else:
        df[col] = df[col].fillna(df[col].median())  


data = df.values

X = data[:, :-1]
y = data[:, -1]
 



split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)

X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

# 2 hidden layers each with 2 neuorons each

input_size = X_train.shape[1]
h1 = 100
h2 = 10
output_size = 1

W1 = np.ones((input_size, h1))
b1 = np.zeros((1, h1))

W2 = np.ones((h1, h2))
b2 = np.zeros((1, h2))

W3 = np.ones((h2, output_size))
b3 = np.zeros((1, output_size))

#forward propagation

def forward(X):
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)

    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    Z3 = A2 @ W3 + b3
    A3 = sigmoid(Z3)

    return A1, A2, A3

#backward propagation
def backward(X, y, A1, A2, A3, lr):
    global W1, b1, W2, b2, W3, b3 

    m = len(y)
    y = y.reshape(-1, 1)

    # Output layer
    dZ3 = A3 - y
    dW3 = (A2.T @ dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    # Layer 2
    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * (A2 * (1 - A2))
    dW2 = (A1.T @ dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # Layer 1
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (A1 * (1 - A1))
    dW1 = (X.T @ dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    W3 -= lr * dW3
    b3 -= lr * db3

#training loop 

lr = 0.01
epochs = 2000

for i in range(epochs):
    A1, A2, A3 = forward(X_train)
    backward(X_train, y_train, A1, A2, A3, lr)

#accuracy 
A1, A2, A3 = forward(X_test)
print("Accuracy:", accuracy(y_test, A3))

