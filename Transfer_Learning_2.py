import numpy as np
import pandas as pd
from skimage.transform import resize
from keras.utils import to_categorical
from keras_vggface.vggface import VGGFace
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import keras

batch_size = 50
num_classes = 5
epochs_top_layers = 5
epochs_all_layers = 5

img_height, img_width = 197, 197

train_dataset = 'train.csv'
eval_dataset = 'val.csv'

base_model = VGGFace(model = 'resnet50', include_top = False, weights = 'vggface',
                     input_shape = (img_height, img_width, 3))

x = base_model.output

x = Flatten()(x)

x = Dense(1024, activation = 'relu')(x)
predictions = Dense(num_classes, activation = 'softmax')(x)

model = Model(inputs = base_model.input, outputs = predictions)

# def preprocess_input(x):
#     x -= 128.8006
#     return x

def get_data(dataset):

    data = pd.read_csv(dataset)
    X = data.drop('target', axis=1) #TODO fix CSV extraction. No labels for TF. use np.genfromtxt
    pixels = np.asarray(X)
    images = np.empty((len(data), img_height, img_width, 3))
    i = 0

    for pixel_sequence in pixels:
        single_image = [float(pixel) for pixel in pixel_sequence]
        single_image = np.asarray(single_image).reshape(126, 126)
        single_image = resize(single_image, (img_height, img_width), order=3,
                              mode='constant')
        ret = np.empty((img_height, img_width, 3))
        ret[:, :, 0] = single_image
        ret[:, :, 1] = single_image
        ret[:, :, 2] = single_image
        images[i, :, :, :] = ret
        i += 1

    labels = to_categorical(data['target'])
    return images, labels

train_x, train_y = get_data(train_dataset)

val_data = get_data(eval_dataset)

train_datagen = ImageDataGenerator(rotation_range=10, shear_range=10, zoom_range=0.1,
                                   fill_mode='reflect', horizontal_flip=True)

train_generator = train_datagen.flow(train_x, train_y, batch_size=batch_size)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer = Adam(lr = 1e-3, beta_1 = 0.9, beta_2 = 0.999,
              epsilon = 1e-08, decay = 0.0), loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

model.fit_generator(generator = train_generator, steps_per_epoch = len(train_x) // batch_size,
                    epochs = epochs_top_layers, validation_data = val_data, verbose=2)

model.compile(optimizer=keras.optimizers.Adadelta(lr=.1, decay=.01),
              loss='categorical_crossentropy', metrics=['accuracy'])


# Early stop and save on each epoch removed. Graphics card lacks memory to run.

model.fit_generator(generator = train_generator, steps_per_epoch = len(train_x) // batch_size,
                    epochs = epochs_all_layers, validation_data = val_data)

#model.save("******.h5")
