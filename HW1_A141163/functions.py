import numpy as np
import pandas as pd

def train_test_split(df , test_size, random_state=None):

    processed_df = df.sample(frac=1, random_state=random_state)

    total_size = len(processed_df)
    split_index = int(total_size * (1 - test_size))

    part1_df = processed_df.iloc[:split_index]
    part2_df = processed_df.iloc[split_index:]

    return part1_df.reset_index(drop=True), part2_df.reset_index(drop=True)

# 1. Define nn layers (initial layer/ activate func: sigmoid and its derivative function)

def init_layer(D, H):
    #D: nums of X_train's col (D = X_train_np.shape[1])
    #H: nums of hidden layers
    rng = np.random.default_rng(42) #make sure the result can be reconstructed
    W = rng.normal(0.0, 0.01, size=(D, H)) #weight
    b = np.zeros((1, H)) #bias
    return W, b

def sigmoid(z): # activation func
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_grad(z): # derivative of sigmoid
    s = sigmoid(z)
    return s * (1.0 - s)

def forward_propagation(X, W0, b0, W1, b1):

    Z = X @ W0 + b0 #before
    A = sigmoid(Z) #after
    cache = (X, Z, A)
    yhat = A @ W1 + b1 # A1 -> y_hat (@ is like np.dot rather than np.multiply)

    return yhat, cache

# 2. Define loss func: Sum-of-Squares Error (SSE) and back prop
def sse_and_grad(A, T):

    diff = A - T
    loss = float(np.sum(diff * diff))   # SSE
    dA = 2.0 * diff                     # gradient wrt A
    return loss, dA

def rms(yhat, y):
    return float(np.sqrt(np.mean((yhat - y)**2)))

def backward_propagation(yhat, y, cache, W0, b0, W1, b1):

    X, Z0, A1 = cache

    # SSE
    loss, dY = sse_and_grad(yhat, y)        # dY = 2*(yhat - y)

    # output layer
    dW1 = A1.T @ dY                          # (H,1)
    db1 = dY.sum(axis=0, keepdims=True)      # (1,1)
    dA1 = dY @ W1.T                          # (N,H)

    # hidden layer
    dZ0 = dA1 * sigmoid_grad(Z0)             # (N,H)
    dW0 = X.T @ dZ0                          # (D,H)
    db0 = dZ0.sum(axis=0, keepdims=True)     # (1,H)
    dX  = dZ0 @ W0.T                         # (N,D)  # optional

    return loss, dW0, db0, dW1, db1, dX

# 3. Define stochastic gradient descent(SGD) algorithm
def sgd_update(W, b, dW, db, lr):
    W -= lr * dW
    b -= lr * db
    return W, b

# ----------------------------- ONLY FOR CLASSIFICATION -----------------------------
def init_layer_cls(D, H):
    rng = np.random.default_rng(42) #make sure the result can be reconstructed
    W = rng.normal(0.0, 0.1, size=(D, H)) #weight
    b = np.zeros((1, H)) #bias
    return W, b

def softmax(z): #activate func of output layer
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward_prop_cls(X, W0, b0, W1, b1):
    # hidden
    Z0 = X @ W0 + b0          
    A1 = sigmoid(Z0)          
    # output
    Z1 = A1 @ W1 + b1         
    yhat = softmax(Z1) 

    cache = (X, Z0, A1, yhat)
    return yhat, cache

def error_rate(yhat, y):
    pred = np.argmax(yhat, axis=1) 

    if y.ndim == 2 and y.shape[1] > 1:  # one hot
        y_true = np.argmax(y, axis=1)
    else:
        y_true = y.reshape(-1)

    return float(np.mean(pred != y_true))

def cross_entropy(yhat, y, eps=1e-8):
    N = y.shape[0]
    yhat = np.clip(yhat, eps, 1.0 - eps)
    loss = -np.sum(y * np.log(yhat)) / N
    dY = (yhat - y) / N
    return loss, dY

def backward_prop_cls(y, cache, W0, W1):
    X, Z0, A1, yhat = cache

    # CE loss + softmax
    loss, dY = cross_entropy(yhat, y)

    # output
    dW1 = A1.T @ dY
    db1 = dY.sum(axis=0, keepdims=True)
    dA1 = dY @ W1.T

    # hidden
    dZ0 = dA1 * sigmoid_grad(Z0)
    dW0 = X.T @ dZ0
    db0 = dZ0.sum(axis=0, keepdims=True)
    dX = dZ0 @ W0.T

    return loss, dW0, db0, dW1, db1, dX
# -------------------------------------------------------------------------------