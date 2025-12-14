import numpy as np
from tensorflow.keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255.0
X_test  = X_test / 255.0


def one_hot(y, C=10):
    v = np.zeros(C)
    v[y] = 1
    return v

def relu(z):
    return np.maximum(0, z)

def drelu(d, z):
    return d * (z > 0)

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / np.sum(e)


def conv_forward(X, W):
    F, kH, kW = W.shape
    H, Wimg = X.shape
    out = np.zeros((F, H-kH+1, Wimg-kW+1))

    for f in range(F):
        for i in range(out.shape[1]):
            for j in range(out.shape[2]):
                out[f, i, j] = np.sum(X[i:i+kH, j:j+kW] * W[f])
    return out

def conv_backward(dZ, X, W):
    F, kH, kW = W.shape
    dW = np.zeros_like(W)

    for f in range(F):
        for i in range(dZ.shape[1]):
            for j in range(dZ.shape[2]):
                dW[f] += dZ[f, i, j] * X[i:i+kH, j:j+kW]
    return dW

def max_pool(X):
    F, H, W = X.shape
    out = np.zeros((F, H//2, W//2))

    for f in range(F):
        for i in range(0, H, 2):
            for j in range(0, W, 2):
                out[f, i//2, j//2] = np.max(X[f, i:i+2, j:j+2])
    return out

def max_pool_backward(dP, X):
    dX = np.zeros_like(X)
    F, H, W = X.shape

    for f in range(F):
        for i in range(dP.shape[1]):
            for j in range(dP.shape[2]):
                region = X[f, i*2:i*2+2, j*2:j*2+2]
                max_val = np.max(region)
                for m in range(2):
                    for n in range(2):
                        if region[m, n] == max_val:
                            dX[f, i*2+m, j*2+n] += dP[f, i, j]
    return dX


def accuracy(X_data, y_data, W1, b1, W2, b2, N=500):
    correct = 0
    for i in range(N):
        Z1 = conv_forward(X_data[i], W1) + b1[:, None, None]
        A1 = relu(Z1)
        P1 = max_pool(A1)
        F  = P1.reshape(-1)
        A2 = softmax(F @ W2 + b2)
        if np.argmax(A2) == y_data[i]:
            correct += 1
    return correct / N


np.random.seed(42)

NUM_FILTERS = 20
W1 = np.random.randn(NUM_FILTERS, 3, 3) * 0.1
b1 = np.zeros(NUM_FILTERS)

F_dim = NUM_FILTERS * 13 * 13
W2 = np.random.randn(F_dim, 10) * 0.1
b2 = np.zeros(10)

lr = 0.01


for step in range(1000):

    X = X_train[step]
    y = one_hot(y_train[step])

    # Forward 
    Z1 = conv_forward(X, W1) + b1[:, None, None]
    A1 = relu(Z1)
    P1 = max_pool(A1)
    F  = P1.reshape(-1)

    Z2 = F @ W2 + b2
    A2 = softmax(Z2)

    loss = -np.sum(y * np.log(A2 + 1e-12))

    # Backward 
    dZ2 = A2 - y
    dW2 = np.outer(F, dZ2)
    db2 = dZ2

    dF  = W2 @ dZ2
    dP1 = dF.reshape(P1.shape)

    dA1 = max_pool_backward(dP1, A1)
    dZ1 = drelu(dA1, Z1)

    dW1 = conv_backward(dZ1, X, W1)
    db1 = np.sum(dZ1, axis=(1,2))

    # Update 
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if step % 100 == 0:
        train_acc = accuracy(X_train, y_train, W1, b1, W2, b2, 300)
        test_acc  = accuracy(X_test, y_test, W1, b1, W2, b2, 300)
        print(f"Step {step} | Loss {loss:.3f} | Train {train_acc:.2f} | Test {test_acc:.2f}")
