import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
import time

start = time.time()

DEV = False
argvs = sys.argv
argc = len(argvs)

if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
  DEV = True

if DEV:
  epochs = 2
else:
  epochs = 30

train_data_path = 'data/train'
validation_data_path = 'data/test'

"""
Параметры
"""
img_width, img_height = 150, 150  # Разрешение картинки
batch_size = 32 
samples_per_epoch = 53  # Кол-во изображений тренирующей выборки
validation_steps = 19  # Кол-во изоюражений тестовой выборки
nb_filters1 = 32  # Кол-во нейронов 1 слой
nb_filters2 = 64  # Кол-во нейронов 2 слой
conv1_size = 3  # Кол-во нейронов 1 свёрточный
conv2_size = 2  # Кол-во нейронов 2 свёрточный
pool_size = 2  # MaxPool размер
classes_num = 3   # Выходной слой \ Кол-во классов
lr = 0.0003  # Точность обучения

model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, border_mode="same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(classes_num, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

"""
Логи
"""
log_dir = 'tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
cbks = [tb_cb]

model.fit_generator(
    train_generator,
    samples_per_epoch=samples_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=cbks,
    validation_steps=validation_steps)

target_dir = 'models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('models/model.h5')
model.save_weights('models/weights.h5')


print(train_generator.class_indices)
