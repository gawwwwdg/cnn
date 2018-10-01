# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 10:00:09 2018

@author: prakh
"""
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
fashion_mnist = keras.datasets.fashion_mnist
import matplotlib.pyplot as plt
import numpy as np



(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','S0andal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
test_images = test_images / 255.0

model =Sequential()
model.add(Convolution2D(32,3,strides=2,padding='valid',input_shape=(28,28,1),activation ='relu',data_format="channels_last"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()

train_images=train_images.reshape(60000,28,28,1)#Reshaping becasue the batch dimension is not included
test_images=test_images.reshape(10000,28,28,1)#Reshaping becasue the batch dimension is not included
model.fit(train_images, train_labels, epochs=10,batch_size=32)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
predictions = model.predict(test_images)




