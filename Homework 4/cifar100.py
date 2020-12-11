## Mark Koszykowski
## ECE472 - Deep Learning
## Assignment 4 - CIFAR100

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler

# Function to change learning rate over epochs
def LR_schedule(epoch):
    lr = .001 * (1 - .02) ** epoch
    return lr

# Initialize necessary params
training_portion = 40000
eps = 1e-10
tot = 0

# Source for manually importing data: https://stackoverflow.com/questions/49045172/cifar10-load-data-takes-long-time-to-download-data
path = 'cifar-100-python/train'
with open(path, 'rb') as file:
    batch = pickle.load(file, encoding='bytes')
    x_train = (batch[b'data'].reshape((len(batch[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)).astype('float32')
    y_train = batch[b'fine_labels']

path = 'cifar-100-python/test'
with open(path, 'rb') as file:
    batch = pickle.load(file, encoding='bytes')
    x_test = (batch[b'data'].reshape((len(batch[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)).astype('float32')
    y_test = batch[b'fine_labels']

y_train = np.array(y_train)
y_test = np.array(y_test)

# Put the image data on some z-scale
mean = np.mean(x_train, axis=(0, 1, 2, 3))
std = np.std(x_train, axis=(0, 1, 2, 3))
x_train = (x_train - mean)/(std + eps)
x_test = (x_test - mean)/(std + eps)

# Divide the provided training set into a validation and training set
x_tune = x_train[training_portion:50000]
y_tune = y_train[training_portion:50000]
x_train = x_train[0:training_portion]
y_train = y_train[0:training_portion]

# Print out the size of the 3 sets
print(f"\nNumber of images in training set {x_train.shape[0]}")
print(f"Number of images in validation set: {x_tune.shape[0]}")
print(f"Number of images in test set: {x_test.shape[0]}\n")

# Source: https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode='nearest',
    cval=0.0,
    horizontal_flip=True,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
    dtype=None
)
datagen.fit(x_train)

# Build the NN model with Convolution, MaxPooling, BN, Dropout, and L2 penalty
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='elu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(.0001), input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='elu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(.0001)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='elu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(.0001)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='elu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(.0001)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation='elu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(.0001)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='elu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(.0001)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, (3, 3), activation='elu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(.0001)))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation='elu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(.0001)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(512, (3, 3), activation='elu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(.0001)))
model.add(BatchNormalization())
model.add(Conv2D(512, (3, 3), activation='elu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(.0001)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.6))

model.add(Flatten())
model.add(Dense(100, activation=tf.nn.softmax))

adam = tf.keras.optimizers.Adam(learning_rate=0.001)

# Train the model
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train, y_train, batch_size=64), steps_per_epoch=x_train.shape[0] // 64,
                    epochs=125, validation_data=(x_tune, y_tune), callbacks=[LearningRateScheduler(LR_schedule)])

# Test model on the test set
probs = model.predict(x_test)

# Calculate top 5 accuracy
top5 = tf.math.in_top_k(y_test, probs, k=5)
for i in range(len(top5)):
    if top5[i] == True:
        tot += 1

top5accuracy = tot/len(top5)
print(top5accuracy)