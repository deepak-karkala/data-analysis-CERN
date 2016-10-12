"""run.py


"""

### Importing libraries
import numpy as np
from costs import *
from helpers import *
from proj1_helpers import *
from toolbox import *

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


def least_squares(y, tx):
    """calculate the least squares solution using normal equation."""
    wopt = np.dot(np.linalg.inv(np.dot(tx.transpose(),tx)),np.dot(tx.transpose(),y))
    mse = sum((y-np.dot(tx,wopt))**2)/(2*len(y))
    return wopt, mse

def least_squares_GD(y, tx, gamma, max_iters):
    """calculate the least squares solution using gradient descent."""
    gradient_losses, gradient_ws = gradient_descent(y, tx, w_initial, max_iters, gamma)



