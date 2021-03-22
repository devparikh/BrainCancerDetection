# importing dependencies
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten, Dropout

train_path = "C:\\Users\\me\\Documents\\BrainCancerDetection\\dataset\\Training"
test_path = "C:\\Users\\me\\Documents\\BrainCancerDetection\\dataset\\Testing"

img_size = 500
batch_size = 32
epochs = 10

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255, 
                                horizontal_flip=True, 
                                rotation_range=20,
                                shear_range=0.4,
                                zoom_range=0.5,
                                width_shift_range=0.15,
                                height_shift_range=0.15)

train_set = train_datagen.flow_from_directory(directory=train_path,
                                                    batch_size=batch_size,
                                                    color_mode = "grayscale",
                                                    shuffle=True,
                                                    target_size=(img_size, img_size))

# scaling only for the testing set
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(directory=test_path,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    color_mode = "grayscale",
                                                    target_size=(img_size, img_size))


# creating the model architecture for brain cancer classification
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(img_size,img_size,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2)))

model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))

model.add(Dense(4))
model.add(Activation("softmax"))

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_set, batch_size=batch_size,epochs=epochs, validation=test_set) 
# saving the model
model.save("model.h5")
