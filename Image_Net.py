from __future__ import print_function
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as backend
import numpy as np
import tensorflow as tf
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import itertools

"""Experimenting with Transfer Learning for CNN's using vvg16"""

vvg16 = VGG16(weights='imagenet', include_top=False,
                         input_shape=(48,48,3))

num_classes = 7
epochs = 10

df = pd.read_csv("sep data.csv")
Y = df.target
X = df.drop("target", axis=1)

N, D = X.shape
# X = X.values
# X = X/255
X = X.values.reshape(N, 48, 48, 3)

# Split in  training set : validation set :  testing set in 80:10:10
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
y_train = (np.arange(num_classes) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(num_classes) == y_test[:, None]).astype(np.float32)

topLayerModel = Sequential()
topLayerModel.add(Dense(256, input_shape=(512,), activation='relu'))
topLayerModel.add(Dense(256, input_shape=(256,), activation='relu'))
topLayerModel.add(Dropout(0.5))
topLayerModel.add(Dense(128, input_shape=(256,), activation='relu'))
topLayerModel.add(Dense(num_classes, activation='softmax'))

topLayerModel.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=.8),
              metrics=["accuracy"])

topLayerModel.fit(X_train, y_train,
          validation_data=(X_test, y_test), verbose=2, epochs=epochs)

inputs = Input(shape=(48, 48, 1))
vg_output = InceptionV3(inputs)
model_predictions = topLayerModel(vg_output)
final_model = Model(input=inputs, output=model_predictions)

y_pred = topLayerModel.predict(X_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(cm)