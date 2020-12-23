#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 01:04:23 2020

@author: elias
"""

#########################################################
#   Name: Elias Zarate
#   Date: 12/10/2020
#   
#########################################################

#import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

# the data, split between train and test sets
from keras.utils import np_utils

import imageio
import numpy as np
from matplotlib import pyplot as plt

# load the model
from keras.models import load_model

# resizing images
from PIL import Image
import os

# the MNIST data is split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape to be samples*pixels*width*height
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# One hot Code
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# convert from integers to floats
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalize to range [0,1]
X_train = X_train / 255.0
X_test = X_test / 255.0


########################Image Resizing###########################
def image_resize():
    # path joining version for other paths
    DIR = './Handwritten_samples/'
    print (len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))
    num_files = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    num_files = num_files -1
    index = 0
    for i in range(num_files): 
        index += 1
        img = Image.open("./Handwritten_samples/Sample" + str(index) + ".png")
        new_img = img.resize((28,28))
        new_img.save("Sample" + str(index) +"_resize.png", "png", optimize=True)    


def create_model():
    print("Estimated time to complete is 10 mins")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
     
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save("test_model.h5")

def evaluate_model():
    # Evaluate model
    print("\nModel evaluation: ")
    model = load_model("test_model.h5")
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    
create_model()
evaluate_model()

model = load_model("test_model.h5")
im = imageio.imread("https://i.imgur.com/a3Rql9C.png") # For testing

gray = np.dot(im[...,:3], [0.299, 0.587, 0.114])
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()

# reshape the image
gray = gray.reshape(1, 28, 28, 1)

# normalize image
gray  = gray /255

# predict digit
prediction = model.predict(gray)
print("\nThe predicted number is: " + str(prediction.argmax()))
print("Actual number is: " + str(5))

for x in range(10):
    results = model.predict(gray)*100
    print(str(x),': ', round(results[0][x],2),'%')


# path joining version for other paths
DIR = './Handwritten_samples/'
print (len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))
num_files = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
num_files = num_files -1
index = 0

# Read through resized files
path = "./Handwritten_resize/"
path = os.path.realpath(path)
os.system(path)
cwd = os.getcwd()

# define the name of the directory to be created
path = "./Handwritten_resize_show"

# Resizing images



def predict_digits():
    count = 0
    for i in range(num_files): 
        count += 1
        # Predicting hand drawn numbers
        im2 = imageio.imread("Sample" + str(count) + "_resize.png")
        gray = np.dot(im2[...,:3], [0.299, 0.587, 0.114])
        plt.imshow(gray, cmap = plt.get_cmap('gray'))
        plt.show()
        # reshape the image
        gray = gray.reshape(1, 28, 28, 1)
        # normalize image
        gray  = gray /255
        # predict digit
        prediction = model.predict(gray)
        print("\nThe predicted number is a " + str(prediction.argmax()))
        results = model.predict(gray)*100
        actual = count -1 # starts at 0
        if actual > 9:
            actual = actual - 10
        print("Actual number is: " + str(actual))
        for x in range(10):
            print(str(x),': ', round(results[0][x],2),'%')
        


print("\nPredicting handwritten numbers")
image_resize()
predict_digits()

print("\n_________________________________________\n")
print("Summary: The model is not entirely accurate in predicting the handwritten digits.,",
      "In fact the accuracy of the model differs from computation to computation which is", 
      "pretty interesting. Some of the possible reasons for such high inaccuracies when predicting",
      "the handwritten digits may include the background color of which the digit was drawn, or the unorthodox style of the handwriting. ",
      "\n\nReferences: https://www.sitepoint.com/keras-digit-recognition-tutorial/")
















