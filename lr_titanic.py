import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(y, y_pred):
    m = len(y)
    return - (1/m) * np.sum(
        y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
    )

def accuracy(y, y_pred):
    y_pred_cls = (y_pred >= 0.5).astype(int)
    return np.mean(y_pred_cls == y)

class LogisticRegression:

    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for i in range(self.epochs):

            t = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(t)

            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

           
    def predict(self, X):
        y_pred = sigmoid(np.dot(X, self.weights) + self.bias)
        return (y_pred >= 0.5).astype(int)
    
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

model = LogisticRegression(lr=0.1, epochs=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy(y_test, model.predict(X_test)))


    
