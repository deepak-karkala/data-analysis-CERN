"""toolbox.py
"""
import numpy as np
from costs import *

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    return -np.dot(tx.transpose(), y-np.dot(tx, w))/len(y)

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    return -np.dot(tx.transpose(),y-np.dot(tx,w))/len(y)

def gradient_descent(y, tx, initial_w, max_iters, gamma): 
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        delL = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * delL
        ws.append(np.copy(w))
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    ws = [initial_w]
    losses = []
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        for n_iter in range(max_iters):
            delL = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            w = w - gamma*delL
            ws.append(np.copy(w))
            losses.append(loss)
            #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            #  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

def least_squares(y, tx):
    """calculate the least squares solution using normal equation."""
    wopt = np.dot(np.linalg.inv(np.dot(tx.transpose(), tx)), np.dot(tx.transpose(), y))
    mse = sum((y-np.dot(tx,wopt))**2)/(2*len(y))
    return mse, wopt

def least_squares_GD(y, tx, gamma, max_iters):
    """calculate the least squares solution using gradient descent."""
    w_initial = np.zeros(tx.shape[1])
    gd_losses, gd_ws = gradient_descent(y, tx, w_initial, max_iters, gamma)
    return gd_losses[-1], gd_ws[-1]

def least_squares_SGD(y, tx, gamma, max_iters):
    """calculate the least squares solution using stochastic gradient descent."""
    w_initial = np.zeros(tx.shape[1])
    batch_size = 1
    sgd_losses, sgd_ws = stochastic_gradient_descent(y, tx, w_initial, batch_size, max_iters, gamma)
    return sgd_losses[-1], sgd_ws[-1]

def ridge_regression(y, tx, lambda_):
    """implements ridge regression."""
    lambdaI = lambda_*np.eye(tx.shape[1])
    wopt = np.dot(np.linalg.inv(np.dot(tx.transpose(),tx) + np.dot(lambdaI.transpose(),lambdaI)),np.dot(tx.transpose(),y))
    mse = sum((y-np.dot(tx,wopt))**2)/(2*len(y))
    return mse, wopt

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    np.random.seed(seed)
    num_train_samples = len(y)*ratio
    indices = np.random.permutation(x.shape[0])
    training_idx, test_idx = indices[:num_train_samples], indices[num_train_samples:]
    train_x, test_x = x[training_idx], x[test_idx]
    train_y, test_y = y[training_idx], y[test_idx]
    return train_x, test_x, train_y, test_y
    
#def logistic_regression(y, tx, gamma, max_iters):

#def reg_logistic_regression(y, tx, gamma, max_iters):


