import sklearn.datasets
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from IPython import display

CONSTANT_VALUE = 5

def add_constant_column(matrix, constant_value):
    # Ensure the input is a NumPy array
    matrix = np.array(matrix)
    # Create a new column filled with the constant value
    constant_column = np.full((matrix.shape[0], 1), constant_value)
    # Add the constant column to the original matrix
    new_matrix = np.hstack((matrix, constant_column))
    return new_matrix

def perceptron(w, x):
    return activation(np.dot(x, w))

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    result_matrix = add_constant_column(np.c_[xx.ravel(), yy.ravel()], 5)
    Z = pred_func(result_matrix)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

def activation(x):
    return np.where( x > 0 , 1, 0)

# Generate a dataset and plot it
np.random.seed(0)
# X, y = sklearn.datasets.make_moons(200, noise=0.20)
X, y = sklearn.datasets.make_blobs(200, centers=2, cluster_std=0.9)

# Add new column filled with constant to existing matrix
X =  add_constant_column(X, 5)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

# initialize weights randomly with mean 0 - [-1; 1]
w = 2*np.random.random((3,)) - 1 

LR = 0.01
iterations = 100

last_error = 0
first_run = True
check_counter = 0
counter_threshould = 5
should_terminate_loop = False

for j in range(iterations):
    # get preds
    pred = perceptron(w, X)
    
    # how much did we miss?
    diff = y - pred
    current_error = np.mean(np.abs(diff))
    
    # 1. Try to tune the learning rate and number of iterations
    # check error before drawing
    # if not first_run:
    #     print('Comparing errors. curent {} and previous {}'.format(current_error, last_error))
    #     if (current_error > last_error):
    #         print("Current error {} is greater than previous error {}".format(current_error, last_error))
    #         should_terminate_loop = True
    #     if (last_error == current_error):
    #         if (check_counter == counter_threshould):
    #             print('Error has not been changed for last {} iterations. Stopping loop...'.format(counter_threshould))
    #             should_terminate_loop = True
    #         check_counter += 1
    #     else: 
    #         check_counter = 0

    if should_terminate_loop or j == LR - 1:
        plot_decision_boundary(lambda x: perceptron(w, x))
        break

    # draw result
    display.clear_output(wait=True)
    # plot_decision_boundary(lambda x: perceptron(w, x))
    display.display("Error:" + str(current_error))
    # time.sleep(0.5)
    
    # update weights
    w = w + LR * np.dot(X.T, diff)
    first_run = False
    last_error = current_error

# FINAL ERROR = 0.015
