import numpy as np

import Logistic_Regression
from load_dataset import load_dataset
import time

tic = time.time()
X_train, Y_train, X_test, Y_test = load_dataset()
toc = time.time()

print("Time taken in loading data = ", (toc-tic)*1000, "ms")

tick = time.time()
results = Logistic_Regression.run_model(X_train,Y_train, X_test, Y_test, 2000, 0.005)
tock = time.time()
print("Time taken in running = ", (tock-tick)*1000, "ms")
print(results)

print("learned parameters--------------------------------------------------------------------------------")
learned_weights = np.load("cat_classifier_lr_weights.npy")
learned_bias = np.load("cat_classifier_lr_bias.npy")
print(learned_weights)
print(learned_bias)