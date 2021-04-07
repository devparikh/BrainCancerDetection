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


classes = ["glioma_tumor",  "meningioma_tumor",  "no_tumor", "pituitary_tumor"]

classes_value = {0: "giloma_tumor",  1: "meningioma_tumor",  2:"no_tumor", 3:"pituitary_tumor"}
labels = []
training_data = []

# printing an image from each class
for category in classes:
    new_train_path = os.path.join(train_path, category)
    # this is used to print all of the different categories the image is in
    class_num = classes.index(category)
    for data in os.listdir(new_train_path):
        train_data = cv2.imread(os.path.join(new_train_path, data))
        # we have a gaussian blur that has a 5x5 gaussian kernel that runs over the image and the SigmaX is calculated from the kernel and the SigmaY is calculated from SigmaX
        train_data = cv2.GaussianBlur(train_data, (5,5), 0)

        train_data = cv2.resize(train_data, (img_size, img_size))
        train_data_gray = cv2.cvtColor(train_data, cv2.COLOR_RGB2GRAY)
        # we are thresholding the image between 50 and 255 so that there is a white segment so that we can see the actually brain mri scan
        ret, train_data_gray  = cv2.threshold(train_data_gray, 50, 255, cv2.THRESH_BINARY)
        # chain approx simple just removes all points that are useless and compresses the contour which saves memory
        contour, hierarchy = cv2.findContours(train_data_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # the -1 will get all of the countours that are in the grayscale image
        # the other parameter is the colour of the line that is created from the contour
        # the 1 which is the last parameter is the thickness of the contour line
        cv2.drawContours(train_data, contour, -1, (255,255,0), 1)
         # converting train_data to grayscale
        train_data = cv2.cvtColor(train_data, cv2.COLOR_RGB2GRAY)

        # here we are checking for the class and if it is a certain class then we do one-hot encoding for it
        if classes.index(category) == 0:
          labels.append([1,0,0,0])
        elif classes.index(category) == 1:
          labels.append([0,1,0,0])
        elif classes.index(category) == 2:
          labels.append([0,0,1,0])
        elif classes.index(category) == 3:
          labels.append([0,0,0,1])
        
        # adding images to training data
        training_data.append(train_data)
        
    print(category)
    cv2_imshow(train_data)
    print(class_num)
    
test_labels = []
testing_data = []
# printing an image from each class
# here I am doing the same preprocessing but for the test set
for category in classes:
    new_test_path = os.path.join(test_path, category)
    class_num = classes.index(category)
    for data in os.listdir(new_test_path):
        test_data = cv2.imread(os.path.join(new_test_path, data))
        # we have a gaussian blur that has a 5x5 gaussian kernel that runs over the image and the SigmaX is calculated from the kernel and the SigmaY is calculated from SigmaX
        test_data = cv2.GaussianBlur(test_data, (5,5), 0)

        test_data = cv2.resize(test_data, (img_size, img_size))
        test_data_gray = cv2.cvtColor(test_data, cv2.COLOR_RGB2GRAY)
        # we are thresholding the image between 50 and 255 so that there is a white segment so that we can see the actually brain mri scan
        ret, test_data_gray  = cv2.threshold(test_data_gray, 50, 255, cv2.THRESH_BINARY)
        # chain approx simple just removes all points that are useless and compresses the contour which saves memory
        contour, hierarchy = cv2.findContours(test_data_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # the -1 will get all of the countours that are in the grayscale image
        # the other parameter is the colour of the line that is created from the contour
        # the 1 which is the last parameter is the thickness of the contour line
        cv2.drawContours(test_data, contour, -1, (255,255,0), 1)
         # converting train_data to grayscale
        test_data = cv2.cvtColor(test_data, cv2.COLOR_RGB2GRAY)

        
        if classes.index(category) == 0:
          test_labels.append([1,0,0,0])
        elif classes.index(category) == 1:
          test_labels.append([0,1,0,0])
        elif classes.index(category) == 2:
          test_labels.append([0,0,1,0])
        elif classes.index(category) == 3:
          test_labels.append([0,0,0,1])

        testing_data.append(test_data)
    
    print(category)
    cv2_imshow(train_data)
    print(class_num)

# converting these lists into numpy arrays
training_data = np.array(training_data)
labels = np.array(labels)

# reshaping the training data so that it can be taken for input by the model
training_data = np.reshape(training_data, (2870, 256,256, 1))



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
model.fit(training_data, labels, batch_size=batch_size,epochs=epochs, validation_data=testing_set) 
# saving the model
model.save("model.h5")
