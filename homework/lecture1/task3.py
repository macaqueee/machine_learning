import numpy as np
import sklearn.datasets
import matplotlib
import matplotlib.pyplot as plt
import time
from IPython import display

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

np.random.seed(0)
# X, y = sklearn.datasets.make_moons(200, noise=0.20)
X, y = sklearn.datasets.make_blobs(200)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

# Initialize weights
w = 2 * np.random.random((2, 3)) - 1

# Define hyperparameters
LR = 1
num_iterations = 10

def activation(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Define prediction function
def predict(X, w):
    return activation(np.dot(X, w))

# Calculate error
def calculate_error(y, pred):
    m = y.shape[0]
    log_likelihood = -np.log(pred[range(m), y])
    return np.sum(log_likelihood) / m

# Update weight
def update_weights(X, y, pred, w, lr):
    m = X.shape[0]
    y_one_hot = np.zeros((m, np.max(y) + 1))
    y_one_hot[range(m), y] = 1
    grad = np.dot(X.T, (pred - y_one_hot)) / m
    w -= lr * grad
    return w

# Training loop
for i in range(num_iterations):
    pred = predict(X, w)
    error = calculate_error(y, pred)
    
    display.display("Error:" + str(error))

    w = update_weights(X, y, pred, w, LR)
    
plot_decision_boundary(lambda x: predict(x, w))
    