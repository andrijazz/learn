import numpy as np


def softmax(action_values, tau=1.0):
    """
    Args:
        action_values (Numpy array): A 2D array of shape (batch_size, num_actions).
                       The action-values computed by an action-value network.
        tau (float): The temperature parameter scalar.
    Returns:
        A 2D array of shape (batch_size, num_actions). Where each column is a probability distribution over
        the actions representing the policy.
    """
    ### START CODE HERE (~2 Lines)
    # Compute the preferences by dividing the action-values by the temperature parameter tau
    preferences = action_values / tau
    # Compute the maximum preference across the actions
    max_preference = np.max(action_values, axis=1) / tau
    ### END CODE HERE

    # Reshape max_preference array which has shape [Batch,] to [Batch, 1]. This allows NumPy broadcasting
    # when subtracting the maximum preference from the preference of each action.
    reshaped_max_preference = max_preference.reshape((-1, 1))

    ### START CODE HERE (~2 Lines)
    # Compute the numerator, i.e., the exponential of the preference - the max preference.
    exp_preferences = np.exp(preferences - reshaped_max_preference)
    # Compute the denominator, i.e., the sum over the numerator along the actions axis.
    sum_of_exp_preferences = np.sum(exp_preferences, axis=1)
    ### END CODE HERE

    # Reshape sum_of_exp_preferences array which has shape [Batch,] to [Batch, 1] to  allow for NumPy broadcasting
    # when dividing the numerator by the denominator.
    reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))

    ### START CODE HERE (~1 Lines)
    # Compute the action probabilities according to the equation in the previous cell.
    action_probs = exp_preferences / reshaped_sum_of_exp_preferences
    ### END CODE HERE

    # squeeze() removes any singleton dimensions. It is used here because this function is used in the
    # agent policy when selecting an action (for which the batch dimension is 1.) As np.random.choice is used in
    # the agent policy and it expects 1D arrays, we need to remove this singleton batch dimension.
    action_probs = action_probs.squeeze()
    return action_probs

