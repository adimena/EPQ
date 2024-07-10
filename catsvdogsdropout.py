import os

import random
from shutil import copyfile

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


if True:
    def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
        all_files = []

        for file_name in os.listdir(SOURCE):
            file_path = SOURCE + file_name

            if os.path.getsize(file_path):
                all_files.append(file_name)
            else:
                print('{} is zero length, so ignoring'.format(file_name))

        n_files = len(all_files)
        split_point = int(n_files * SPLIT_SIZE)

        shuffled = random.sample(all_files, n_files)

        train_set = shuffled[:split_point]
        test_set = shuffled[split_point:]

        for file_name in train_set:
            copyfile(SOURCE + file_name, TRAINING + file_name)

        for file_name in test_set:
            copyfile(SOURCE + file_name, TESTING + file_name)


    CAT_SOURCE_DIR = "./PetImages/Cat/"
    TRAINING_CATS_DIR = "./cats-v-dogs/training/cats/"
    TESTING_CATS_DIR = "./cats-v-dogs/testing/cats/"
    DOG_SOURCE_DIR = "./PetImages/Dog/"
    TRAINING_DOGS_DIR = "./cats-v-dogs/training/dogs/"
    TESTING_DOGS_DIR = "./cats-v-dogs/testing/dogs/"

    ROWS = 150
    COLS = 150
    CHANNELS = 3
    
    split_size = .9
    split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
    split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)
     
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(ROWS, COLS, CHANNELS), activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, (1,1), activation='relu'))
#model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dropout(0.4))

model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=2, activation='sigmoid'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc'])


TRAINING_DIR = './cats-v-dogs/training'
train_datagen = ImageDataGenerator(
        rescale=1 / 255,
        rotation_range=40,
        width_shift_range=.2,
        height_shift_range=.2,
        shear_range=.2,
        zoom_range=.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        batch_size=64,
        class_mode='binary',
        target_size=(150, 150)
    )

VALIDATION_DIR = './cats-v-dogs/testing'
validation_datagen = ImageDataGenerator(
        rescale=1 / 255,
        rotation_range=40,
        width_shift_range=.2,
        height_shift_range=.2,
        shear_range=.2,
        zoom_range=.2,
        horizontal_flip=True,
        fill_mode='nearest'

    )
validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        batch_size=64,
        class_mode='binary',
        target_size=(150, 150)
    )


import warnings
from PIL import Image as pil_image

#warnings.filterwarnings('ignore')
#print('warnings ignored')

history = model.fit(train_generator,
                    epochs=15,
                    verbose=1,
                    validation_data=validation_generator)

model.save('./cnn6.keras')
