# importing dependencies
import tensorflow as tf
import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten, Dropout

img_size = 256
epochs = 10
BATCH_SIZE = 32

train_path = "//content//Training"
test_path = "//content//Testing"

train_datagen = ImageDataGenerator(rescale=1./255, 
                                horizontal_flip=True,
                                vertical_flip=True, 
                                rotation_range=20,
                                shear_range=0.4,
                                zoom_range=0.7,
                                width_shift_range=0.2,
                                height_shift_range=0.2)

training_set = train_datagen.flow_from_directory(directory=train_path,
                                                    batch_size=BATCH_SIZE,
                                                    color_mode = "grayscale",
                                                    classes = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"],
                                                    shuffle=True,
                                                    target_size=(img_size, img_size))

# scaling only for the testing set
test_datagen = ImageDataGenerator(rescale=1./255)

testing_set = test_datagen.flow_from_directory(directory=test_path,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    classes = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"],
                                                    color_mode = "grayscale",
                                                    target_size=(img_size, img_size))

    
classes = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
# printing an image from each class
for category in classes:
    new_train_path = os.path.join(train_path, category)

    for data in os.listdir(new_train_path):
        train_data = cv2.imread(os.path.join(new_train_path, data))
        
      
        train_data = cv2.resize(train_data, (img_size, img_size))
        
         # we have a gaussian blur that has a 5x5 gaussian kernel that runs over the image and the SigmaX is calculated from the kernel and the SigmaY is calculated from SigmaX
        train_data = cv2.GaussianBlur(train_data, (5,5), 0)
        # converting to grayscale
        train_data_gray = cv2.cvtColor(train_data, cv2.COLOR_RGB2GRAY)
        
        # we are thresholding the image between 50 and 255 so that there is a white segment so that we can see the actual brain mri scan
        ret, train_data_gray = cv2.threshold(train_data_gray, 50, 255, cv2.THRESH_BINARY)
        
        # chain approx simple just removes all points that are useless and compresses the contour which saves memory
        # a hierarchy is when contours have relationships with other contours generally it is the relationship between the parent contour(outer contour) and the child contour(the inner contour)
        contour, hierarchy = cv2.findContours(train_data_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # the -1 will get all of the countours that are in the grayscale image
        # the other parameter is the colour of the line that is created from the contour
        # the 1 which is the last parameter is the thickness of the contour line
        cv2.drawContours(train_data, contour, -1, (255, 0, 0), 1)


    print(category)
    cv2_imshow(train_data)

# scaling only for the testing set
test_datagen = ImageDataGenerator(rescale=1./255)

testing_set = test_datagen.flow_from_directory(directory=test_path,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    color_mode = "grayscale",
                                                    target_size=(img_size, img_size))


# creating the model architecture for brain cancer classification
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(img_size,img_size,1)))
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

# 1e-5 is the learning rate for the model
model.compile(optimizer=Adam(1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Training my model
model.fit(training_set, batch_size=batch_size,epochs=epochs, validation_data=testing_set) 
# saving the model
model.save("model.h5")
