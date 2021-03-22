# importing dependencies
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten, Dropout

train_path = "C:\\Users\\me\\Documents\\BrainCancerDetection\\dataset\\Training"
test_path = "C:\\Users\\me\\Documents\\BrainCancerDetection\\dataset\\Testing"

img_size = 500
batch_size = 32
epochs = 10

classes = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
# printing an image from each class
for category in classes:
    new_train_path = os.path.join(train_path, category)

    for data in os.listdir(new_train_path):
        train_data = cv2.imread(os.path.join(new_train_path, data))
    
    print(category)
    cv2.imshow(data)
   
# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255, 
                                horizontal_flip=True, 
                                rotation_range=20,
                                shear_range=0.4,
                                zoom_range=0.5,
                                width_shift_range=0.15,
                                height_shift_range=0.15)

training_set = train_datagen.flow_from_directory(directory=train_path,
                                                    batch_size=batch_size,
                                                    color_mode = "grayscale",
                                                    shuffle=True,
                                                    target_size=(img_size, img_size))

# scaling only for the testing set
test_datagen = ImageDataGenerator(rescale=1./255)

testing_set = test_datagen.flow_from_directory(directory=test_path,
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

model.fit(training_set, batch_size=batch_size,epochs=epochs, validation=testing_set) 
# saving the model
model.save("model.h5")
