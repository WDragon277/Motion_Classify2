
import matplotlib as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import pandas as pd
import glob
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .repository import dataset

def make_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.summary()

    # 4. Dense 층 추가하기
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.summary()

    # 5. 모델 컴파일하기
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 6. 훈련하기
    model.fit(train_images, train_labels, epochs=5)

    # 7. 모델 평가하기
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)

    return model