import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import time


# Описание путей
model_path = 'models/model.h5'
model_weights_path = 'models/weights.h5'
test_path = 'data/alien_test'

# Загрузка моделей
model = load_model(model_path)
model.load_weights(model_weights_path)

# Описание разрешения изображения
img_width, img_height = 150, 150


# Функция предсказания
def predict(file):
    x = load_img(file, target_size=(img_width, img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    # print(result)
    answer = np.argmax(result)
    if answer == 1:
        print("Predicted: банан")
    elif answer == 0:
        print("Predicted: банка")
    elif answer == 2:
        print("Predicted: бандерольный конверт")

    return answer


# Проход по изображениям из test_path
for _, ret in enumerate(os.walk(test_path)):
    for i, filename in enumerate(ret[2]):
        if filename.startswith("."):
            continue

        print(ret[0] + '/' + filename)
        result = predict(ret[0] + '/' + filename)
        print(" ")
