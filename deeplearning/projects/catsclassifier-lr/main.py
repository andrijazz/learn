import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


def load_dataset():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_X = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_y = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_X = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_y = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_y = train_y.reshape((train_y.shape[0], 1))
    test_y = test_y.reshape((test_y.shape[0], 1))

    return train_X, train_y, test_X, test_y, classes


def plot_sample(X, y, sample):
    plt.imshow(X[sample])
    plt.show()
    print("y = {} , it's a {} picture".format(str(y[sample][0]), classes[np.squeeze(y[sample][0])].decode("utf-8")))


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    w = np.zeros((dim, 1))
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


def propagate(W, b, X, y):
    m = X.shape[0]
    h = sigmoid(np.dot(X, W) + b)
    J = -(1.0 / m) * np.sum(y * np.log(h) - (1 - y) * np.log(1 - h))
    dW = np.dot((h-y).T, X).T / m
    db = np.sum(h - y) / m

    assert (dW.shape == W.shape)
    assert (db.dtype == float)
    cost = np.squeeze(J)
    assert (cost.shape == ())

    grads = {"dW": dW,
             "db": db}

    return grads, cost


def optimize(W, b, X, Y, num_iterations, learning_rate, print_cost=True):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    W -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []

    for i in range(num_iterations):

        ### START CODE HERE ###
        grads, cost = propagate(W, b, X, Y)
        ### END CODE HERE ###

        # Retrieve derivatives from grads
        dW = grads["dW"]
        db = grads["db"]

        ### START CODE HERE ###
        W = W - learning_rate * dW
        b = b - learning_rate * db
        ### END CODE HERE ###

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration {}: {}".format(i, cost))

    params = {"W": W,
              "b": b}

    grads = {"dW": dW,
             "db": db}

    return params, grads, costs


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

    for i in range(A.shape[0]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[i, 0] > 0.5:
            Y_prediction[i, 0] = 1
        else:
            Y_prediction[i, 0] = 0
        pass
        ### END CODE HERE ###

    assert (Y_prediction.shape == (m, 1))

    return Y_prediction

# load
train_X, train_y, test_X, test_y, classes = load_dataset()

# plot
plot_sample(train_X, train_y, 25)

# reshape
m_train = train_X.shape[0]
# We could have also written train_X_flatten = train_X.reshape(train_X.shape[0], -1)
train_X_flatten = train_X.reshape(m_train, train_X.shape[1] * train_X.shape[2] * train_X.shape[3])
m_test = test_X.shape[0]
test_X_flatten = test_X.reshape(m_test, test_X.shape[1] * test_X.shape[2] * test_X.shape[3])
n = train_X_flatten.shape[1]

# normalize
train_X_normalized = train_X_flatten.astype(np.float32) / 255
test_X_normalized = test_X_flatten.astype(np.float32) / 255

W, b = initialize_with_zeros(n)

params, grads, costs = optimize(W, b, train_X_normalized, train_y, num_iterations=10000, learning_rate=0.0005)
np.save('weights', params)

predicted_test_y = predict(params["W"], params["b"], test_X_normalized)
print("Accuracy = {}".format(np.sum(predicted_test_y == test_y).astype(np.float32)/test_y.shape[0]))
