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



def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper
precision = as_keras_metric(tf.metrics.precision)


num_classes = 7
batch_size = 300
epochs = 150
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

#TODO replace with np.genfromtxt
df = pd.read_csv("sep data.csv")
Y = df.target
X = df.drop("target", axis=1)


# keras with tensorflow backend
N, D = X.shape
# X = X.values
# X = X/255
X = X.values.reshape(N, 48, 48, 1)

# Split in  training set : validation set :  testing set in 80:10:10
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=0)
y_train = (np.arange(num_classes) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(num_classes) == y_test[:, None]).astype(np.float32)

checkpoint = ModelCheckpoint('best_weights.hdf5', monitor='val_loss',
                             save_best_only=True)
callbacks = [checkpoint]


model = Sequential()

# 1 - Convolution
model.add(Conv2D(64, (5, 5), padding='same', input_shape=(48, 48, 1)))
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

# Flattening
model.add(Flatten())

# Fully connected layer 1st layer
model.add(Dense(512, kernel_regularizer=l2(0.01)))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

# Fully connected layer 2nd layer
model.add(Dense(256, kernel_regularizer=l2(0.01)))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=1),
              metrics=["accuracy", precision])

model.fit(X_train, y_train,
          batch_size=batch_size,
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

# model.save("cnn_model.h5")


plt.figure()
plot_confusion_matrix(cm, classes=label_map, normalize=True, title="normalized matrix")
plt.show()

model.save("ConanCNN.h5")