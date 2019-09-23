import numpy as np
import scipy.misc
import matplotlib.pyplot as plt


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    m = X.shape[0]
    Y_prediction = np.zeros((m, 1))

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(X, w) + b)
    ### END CODE HERE ###

    # for i in range(A.shape[0]):
    #
    #     # Convert probabilities A[0,i] to actual predictions p[0,i]
    #     if A[i, 0] > 0.5:
    #         Y_prediction[i, 0] = 1
    #     else:
    #         Y_prediction[i, 0] = 0
    #     pass
    #     ### END CODE HERE ###
    #
    # assert (Y_prediction.shape == (m, 1))

    # return Y_prediction
    return A

# load params
params = np.load('weights.npy')
W = params.item(0)["W"]
b = params.item(0)["b"]

# load image
im = scipy.misc.imread("test4.jpeg")

# resize image
resized_im = scipy.misc.imresize(im, (64, 64), interp="bilinear")

plt.imshow(resized_im)
plt.show()

# reshape image img
input = np.reshape(resized_im, (1, 64 * 64 * 3)).astype(np.float32) / 255
output = predict(W, b, input)
print(output[0])
