import numpy as np
from tensorflow.keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0


def one_hot(y, num_classes=10):
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1
    return out

y_train_oh = one_hot(y_train, 10)
y_test_oh = one_hot(y_test, 10)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


input_size = 784
h1 = 150
h2 = 50
output_size = 10

np.random.seed(1)

W1 = np.random.randn(input_size, h1) * np.sqrt(1 / input_size)
b1 = np.zeros((1, h1))

W2 = np.random.randn(h1, h2) * np.sqrt(1 / h1)
b2 = np.zeros((1, h2))

W3 = np.random.randn(h2, output_size) * np.sqrt(1 / h2)
b3 = np.zeros((1, output_size))

def forward(X):
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)

    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)

    return A1, A2, A3

def backward(X, y, A1, A2, A3, lr):
    global W1, b1, W2, b2, W3, b3

    m = len(X)

    
    dZ3 = A3 - y   
    dW3 = (A2.T @ dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    
    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * (A2 * (1 - A2))
    dW2 = (A1.T @ dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (A1 * (1 - A1))
    dW1 = (X.T @ dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    W3 -= lr * dW3
    b3 -= lr * db3

#training
lr = 0.01
epochs = 100
batch_size = 128
n_batches = len(X_train) // batch_size

for epoch in range(epochs):
    for i in range(n_batches):
        Xb = X_train[i*batch_size:(i+1)*batch_size]
        yb = y_train_oh[i*batch_size:(i+1)*batch_size]

        A1, A2, A3 = forward(Xb)
        backward(Xb, yb, A1, A2, A3, lr)

    
    _, _, A3_train = forward(X_train[:3000])
    preds = np.argmax(A3_train, axis=1)
    acc = np.mean(preds == y_train[:3000])
    print(f"Epoch {epoch} | Train Accuracy: {acc:.4f}")


_, _, A3_test = forward(X_test)
y_pred = np.argmax(A3_test, axis=1)
acc_test = np.mean(y_pred == y_test)

print("\nFINAL TEST ACCURACY:", acc_test)

