import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D,Dropout

def new_cnn(in_shape):
    #create model
    model = Sequential()
    #add model layers
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=in_shape))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    #model.add(Conv2D(128, kernel_size=1, activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

