import numpy as np
import copy

def sigmoid(z):
    s = (np.exp(z * -1) + 1) ** -1
    # s = 1 / (1 + np.exp(-z))
    return s

def initialize_weights(feature_count):
    weights = np.zeros(shape=(feature_count, 1))
    bias = 0
    return weights, bias

def propagate(X, Y, weights, bias):

    # Total number of training examples = m.
    m = X.shape[1]

    # Forward propagation:

    hypothesis = np.dot(weights.T, X) + bias
    activation = sigmoid(hypothesis)
    total_cost = -1 * np.sum(np.dot(Y, np.log(activation).T) + np.dot((1 - Y), np.log(1 - activation).T)) / m
    #total_cost = np.squeeze(np.array(total_cost))

    # Backward propagation:
    delta_weight = (np.dot(X, (activation - Y).T)) / m
    delta_bias = (np.sum(activation - Y)) / m

    gradients = {
                    "dw": delta_weight,
                    "db": delta_bias
                }
    return total_cost, gradients


def train_model(weights, bias, X, Y, number_iterations, learning_rate):

    weights = copy.deepcopy(weights)
    bias = copy.deepcopy(bias)

    costs = []

    for i in range(number_iterations):
        total_cost, gradients = propagate(X, Y, weights, bias)

        dw = gradients["dw"]
        db = gradients["db"]

        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db
        costs.append(total_cost)
        if i % 100 == 0:
            print(type(total_cost))
            print(" Total cost after 100 iterations = ", total_cost)

    learned_parameters = {
        "w": weights,
        "b": bias}

    final_gradients = {
        "dw": dw,
        "db": db}

    return learned_parameters, final_gradients, costs


def predict(weights, bias, X):
    m = X.shape[1]
    Y_predictions = np.zeros((1, m))

    weights = weights.reshape(X.shape[0], 1)
    activations = sigmoid(np.dot(weights.T, X) + bias)
    print(activations.shape)

    for i in range(activations.shape[1]):
        Y_predictions[0, i] = 1 if activations[0, i] > 0.5 else 0

    return Y_predictions

def run_model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate):
    weights, bias = initialize_weights(X_train.shape[0])

    learned_parameters, final_gradients, costs = train_model(weights, bias, X_train, Y_train, num_iterations, learning_rate)

    weights = learned_parameters["w"]
    bias = learned_parameters["b"]

    np.save("cat_classifier_lr_weights.npy", weights)
    np.save("cat_classifier_lr_bias.npy", bias)

    Y_train_predictions = predict(weights, bias, X_train)
    Y_test_predictions = predict(weights, bias, X_test)

    train_set_accuracy = (100 - np.mean(np.abs(Y_train_predictions - Y_train)) * 100)
    test_set_accuracy = (100 - np.mean(np.abs(Y_test_predictions - Y_test)) * 100)

    print(costs)

    results = {
        "costs": costs,
        "training_set_predictions": Y_train_predictions,
        "test_set_predictions": Y_test_predictions,
        "train_set_accuracy": train_set_accuracy,
        "test_set_accuracy": test_set_accuracy
    }

    return results
