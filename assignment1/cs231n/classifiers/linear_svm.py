import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  
  ### Raizo
  # For the cost function, we go for each label that is not the correct one, and compute the
  # difference between it and the correct one. Then using the formula max(0, s_non - s_correct + 1),
  # thus: if the s_correct is greater than the other labels in at least 1, it will produce a cost
  # of 0, implying that the loss function will lead the right label value to +infinity and the other ones

  # if it's found a W that produces a Loss=0, then 2W produces a Loss=0
  # to -inifity
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      # Remember to calculate the derivative of W'(j).Xi and -W'(y[i]).Xi
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # derivative of W'(j).Xi
        dW[:,j] += X[i] # No need to use X[i].T broadcasting does it for you
        # derivative of -W'(y[i]).Xi
        dW[:,y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * np.sum(W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero (D,C)
  num_train = X.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W) # (N, C)
  correct = scores[xrange(num_train), y]
  # new axis add a dimension, can transfor an array to a vector or a column
  margin = np.maximum(0, scores - correct[:,np.newaxis] + 1)
  margin[range(num_train), y] = 0

  loss = 1/num_train * np.sum(margin.sum(axis=1))
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  dmargin = (margin > 0) * 1
  counter = np.sum(dmargin, axis=1) # array
  # advance indexing should always be done usig arrays
  # dW is incremented with 1 Xi for i!=yi, and -count() Xj for i==yi and j!=yj
  dmargin[range(num_train), y] = -counter
  dW = X.T.dot(dmargin)
  
  dW /= num_train
  dW += reg * 2 * W
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
