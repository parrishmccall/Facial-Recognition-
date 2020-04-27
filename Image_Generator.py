from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
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
from keras.models import load_model
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras import optimizers

""" Experimenting with the Keras CNN Image Generator"""

num_classes = 5
batch_size = 200
epochs = 100
label_map = ['Andy', 'Australian', 'Conan', 'Daniel', 'Hugh']

df = pd.read_csv("combinedmaster target.csv")
Y = df.target
X = df.drop("target", axis=1)


# keras with tensorflow backend
N, D = X.shape
# # X = X.values
# # X = X/255
X = X.values.reshape(N, 126, 126, 1)

# Split in  training set : validation set :  testing set in 80:10:10
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)
y_train = (np.arange(num_classes) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(num_classes) == y_test[:, None]).astype(np.float32)

# checkpoint = ModelCheckpoint('best_weights.hdf5', monitor='val_loss',
#                              save_best_only=True)
# callbacks = [checkpoint]

train_datagen = ImageDataGenerator(
    rotation_range = 10,
    shear_range = 10, # 10 degrees
    zoom_range= 0.1,
    fill_mode= 'reflect',
    horizontal_flip=True)

train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size  = batch_size)

model = Sequential()

# 1 - Convolution
model.add(Conv2D(64, (5, 5), padding='same', input_shape=(126, 126, 1)))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

# 2nd Convolution layer
model.add(Conv2D(128, (5, 5), padding='same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

# 3rd Convolution layer
model.add(Conv2D(256, (3, 3), padding='same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

# 4th Convolution layer
model.add(Conv2D(512, (3, 3), padding='same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

# Flattening
model.add(Flatten())

# Fully connected layer 1st layer
model.add(Dense(512))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

# Fully connected layer 2nd layer
model.add(Dense(256))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(num_classes, activation='softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=2),
              metrics=["accuracy"])

model.fit_generator(generator=train_generator,
          epochs=epochs,
          verbose=2,
          validation_data=(X_test, y_test), shuffle=True)


score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(cm)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

#model.save("cnn_model_large.h5")


plt.figure()
plot_confusion_matrix(cm, classes=label_map, normalize=True, title="normalized matrix")
plt.show()
