import numpy as np

# =====================
# Load MNIST locally
# =====================
data = np.load("mnist.npz")
x_train = data["x_train"][:2000]
y_train = data["y_train"][:2000]

X = x_train.reshape(-1, 1, 28, 28) / 255.0
Y = np.eye(10)[y_train]


# ============================================
# im2col (CORRECT VERSION)
# ============================================
def im2col(X, FH, FW):
    N, C, H, W = X.shape
    out_h = H - FH + 1
    out_w = W - FW + 1
    
    col = np.zeros((N, out_h, out_w, C, FH, FW))

    for i in range(out_h):
        for j in range(out_w):
            col[:, i, j, :, :, :] = X[:, :, i:i+FH, j:j+FW]
    
    return col.reshape(N * out_h * out_w, -1), out_h, out_w


# ============================================
# Convolution Layer
# ============================================
class Conv:
    def __init__(self, in_ch, out_ch, k):
        self.FH = self.FW = k
        self.W = np.random.randn(out_ch, in_ch, k, k) * 0.01
        self.b = np.zeros((out_ch,))
    
    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        
        col, oh, ow = im2col(X, self.FH, self.FW)
        W_col = self.W.reshape(self.W.shape[0], -1)

        out = col @ W_col.T + self.b
        out = out.reshape(N, oh, ow, -1).transpose(0, 3, 1, 2)

        self.col = col
        self.oh, self.ow = oh, ow
        return out
    
    def backward(self, d_out, lr):
        N, F, oh, ow = d_out.shape
        d_out_r = d_out.transpose(0, 2, 3, 1).reshape(N * oh * ow, F)

        dW = d_out_r.T @ self.col
        db = d_out_r.sum(axis=0)

        self.W -= lr * dW.reshape(self.W.shape)
        self.b -= lr * db


# ============================================
# MaxPool 2×2
# ============================================
class MaxPool:
    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape

        out_h, out_w = H // 2, W // 2
        Xr = X.reshape(N, C, out_h, 2, out_w, 2)

        self.mask = (Xr == Xr.max(axis=(3, 5), keepdims=True))
        return Xr.max(axis=(3, 5))

    def backward(self, d_out):
        N, C, H, W = self.X.shape
        out_h, out_w = H // 2, W // 2

        d = d_out[:, :, :, None, :, None]
        d = d * self.mask

        return d.reshape(N, C, H, W)


# ============================================
# Fully Connected Layer
# ============================================
class Dense:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * 0.01
        self.b = np.zeros((1, out_dim))
    
    def forward(self, X):
        self.X = X
        return X @ self.W + self.b
    
    def backward(self, d_out, lr):
        dW = self.X.T @ d_out
        db = d_out.sum(axis=0, keepdims=True)
        dX = d_out @ self.W.T

        self.W -= lr * dW
        self.b -= lr * db

        return dX


# ============================================
# Activation + Softmax
# ============================================
def relu(x): return np.maximum(0, x)
def relu_back(d, x): return d * (x > 0)
def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)
def loss_fn(p, y):
    return -np.mean(np.sum(y * np.log(p + 1e-12), axis=1))


# ============================================
# Build Model
# ============================================
conv = Conv(1, 8, 3)
pool = MaxPool()
dense = Dense(8 * 13 * 13, 10)

lr = 0.01
epochs = 150
batch = 64

# ============================================
# Training Loop
# ============================================
for epoch in range(epochs):
    idx = np.random.permutation(len(X))
    Xsh, Ysh = X[idx], Y[idx]
    
    losses = []
    for i in range(0, len(X), batch):
        xb = Xsh[i:i+batch]
        yb = Ysh[i:i+batch]

        # Forward
        z1 = conv.forward(xb)
        a1 = relu(z1)
        p1 = pool.forward(a1)
        flat = p1.reshape(len(xb), -1)
        logits = dense.forward(flat)
        probs = softmax(logits)

        loss = loss_fn(probs, yb)
        losses.append(loss)

        # BACKPROP
        dlog = (probs - yb) / len(xb)
        dflat = dense.backward(dlog, lr)
        dp1 = dflat.reshape(p1.shape)
        da1 = pool.backward(dp1)
        dz1 = relu_back(da1, z1)
        conv.backward(dz1, lr)

    print(f"Epoch {epoch+1}, Loss = {np.mean(losses):.4f}")

# ============================================
# Test Accuracy
# ============================================
logits = dense.forward(pool.forward(relu(conv.forward(X))).reshape(len(X), -1))
preds = np.argmax(logits, axis=1)
print("Accuracy:", np.mean(preds == y_train)*100, "%")
