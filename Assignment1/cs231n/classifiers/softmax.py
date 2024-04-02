from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    y_pred = X.dot(W)
    after_exp = np.exp(y_pred)

    after_nor = np.zeros(y_pred.shape)
    for i in range(len(y_pred)):
        s = np.sum(after_exp[i])
        for j in range(len(y_pred[0])):
            after_nor[i][j] = after_exp[i][j] / s

    loss = np.zeros(len(y_pred))
    for i in range(len(loss)):
        loss[i] = -np.log(after_nor[i][y[i]])
    loss = sum(loss) / len(loss)
    loss += reg * np.sum(np.square(W))

    for i in range(len(y)):
        for j in range(len(W[0])):
            if j == y[i]:
                dW[:, j] += -X[i] + X[i] * after_nor[i][j]  # 当预测极为接近标签时，after_nor[i][j]接近1，对应dW更新很小
            else:
                dW[:, j] += X[i] * after_nor[i][j]  # 预测偏差越大，更新越大

    dW /= len(y)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    y_pred = X.dot(W)
    after_exp = np.exp(y_pred)
    after_nor = after_exp / np.sum(after_exp, axis=1, keepdims=True)
    y_true_classes = np.zeros_like(y_pred)
    y_true_classes[range(len(y)), y] = 1.0
    loss = -np.sum(y_true_classes * np.log(after_nor)) / len(y) + reg * np.sum(np.square(W))

    dW = -np.dot(X.T, y_true_classes - after_nor) / len(y) + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
