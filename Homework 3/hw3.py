## Mark Koszykowski
## ECE472 - Deep Learning
## Assignment 3

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

# Initialize necessary params
tot = 0
training_portion = 50000

# Retrieve training and testing sets from file
with np.load('mnist.npz', allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

# Divide the provided training set into a validation and training set
x_tune = x_train[training_portion:60000]
y_tune = y_train[training_portion:60000]
x_train = x_train[0:training_portion]
y_train = y_train[0:training_portion]

# Make sure image data in type 'float32' so TF can use it
x_train = x_train.astype('float32')
x_tune = x_tune.astype('float32')
x_test = x_test.astype('float32')


# Normalize RGB values
x_train /= 255
x_tune /= 255
x_test /= 255

# Print out the size of the 3 sets
print(f"\nNumber of images in training set {x_train.shape[0]}")
print(f"Number of images in validation set: {x_tune.shape[0]}")
print(f"Number of images in test set: {x_test.shape[0]}\n")

# Build the NN model with Dropout and L2 penalty
model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(10, kernel_regularizer='l2', activation=tf.nn.softmax))

# Train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=x_train, y=y_train, epochs=10, validation_data=(x_tune, y_tune))

x = np.linspace(1, len(history.history['loss']))

plot(x, history.history['loss'])

# Create an array of predictions for the test set
probs = model.predict(x_test)
preds = np.zeros(10000, dtype=int)
for i in range(10000):
    preds[i] = np.argmax(probs[i])

# Calculate the accuracy
for i in range(10000):
    if preds[i] == y_test[i]:
       tot += 1

correct = tot/10000 * 100
print(f"\nTest set accuracy: {correct}%")