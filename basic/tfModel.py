import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from random import random

def generate_dataset(num_samples, test_size=0.33):
    """Generates train/test data for sum operation
    :param num_samples (int): Num of total samples in dataset
    :param test_size (int): Ratio of num_samples used as test set
    :return x_train (ndarray): 2d array with input data for training
    :return x_test (ndarray): 2d array with input data for testing
    :return y_train (ndarray): 2d array with target data for training
    :return y_test (ndarray): 2d array with target data for testing
    """

    # build inputs/targets for sum operation: y[0][0] = x[0][0] + x[0][1]
    x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([[i[0] + i[1]] for i in x])

    # split dataset into test and training sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = generate_dataset(2000)
    
#     building the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(5, input_dim=2, activation='sigmoid'),
        tf.keras.layers.Dense(1,  activation='sigmoid'),
    ])
    
#     optimizer 
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
#     compiling the model
    model.compile(optimizer=optimizer,loss='mse')
#     model training 
    history = model.fit(x_train,y_train,epochs=100)
    
#   model evaluation
    model.evaluate(x_test, y_test)
    
#     predict model
    data=[[0.2,0.3],[1.3,0.5]]
    predictions = model.predict(data)
    
for d, p  in zip(data, predictions):
    print("\nPredictions :\n")
    print("{} + {} = {}".format(d[0],d[1],p[0]))    