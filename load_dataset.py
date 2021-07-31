import numpy as np
import glob
from PIL import Image
from random import shuffle

def load_images():
    dataset = []
    for img in glob.glob("cats_dataset\*.JPG"):
        img = Image.open(img)
        img_arr = np.array(img)
        dataset.append([img_arr, 1])

    for img in glob.glob("non-cat_dataset\*.JPG"):
        img = Image.open(img)
        img_arr = np.array(img)
        dataset.append([img_arr, 0])

    shuffle(dataset)
    return dataset

def load_dataset():
    dataset = load_images()
    X_train = []
    Y_train = []
    for i in range(600):
        X_train.append(dataset[i][0])
        Y_train.append(dataset[i][1])

    X_test = []
    Y_test = []
    for j in range(34):
        X_test.append(dataset[600+j][0])
        Y_test.append(dataset[600+j][1])

    X_train = np.array(X_train).reshape(600, -1).T
    Y_train = np.array(Y_train).reshape(600,1).T

    X_test = np.array(X_test).reshape(34, -1).T
    Y_test = np.array(Y_test).reshape(34,1).T

    X_train = X_train / 255
    X_test = X_test / 255

    return X_train, Y_train, X_test, Y_test


a,b,c,d = load_dataset()

print(a.shape)
print(b.shape)
print(c.shape)
print(d.shape)












