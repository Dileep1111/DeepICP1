import pandas
import keras.models
from keras.layers.core import Dense, Activation

# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
dataset = pd.read_csv("boston.csv", header=None).values
# print(dataset)
import numpy as np
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,0:13], dataset[:,13],
                                                    test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nn = keras.models.Sequential() # create model
my_first_nn.add(Dense(20, input_dim=13, activation='relu'))# hidden layer
my_first_nn.add(Dense(40, activation='relu'))
my_first_nn.add(Dense(80, activation='relu'))
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='mean_squared_error', optimizer='adam')
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, verbose=0,
                                     initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test, verbose=0))
