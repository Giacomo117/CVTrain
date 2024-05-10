import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random #to avoid overfitting
#for the CNN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#read all the image in the dataset
Datadirectory = "mrlEyeDataset2/"
Classes = ["Closed_Eyes", "Open_Eyes"]
for category in Classes:
    path = os.path.join(Datadirectory, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        plt.imshow(img_array, cmap="gray")
        plt.show()
        break
    break

#resize image for feature extraction
img_size = 224
new_array = cv2.resize(backtorgb, (img_size, img_size))
plt.imshow(new_array, cmap="gray")
plt.show()


#create training data
training_Data = []
def create_training_Data():
   for category in Classes:
       path = os.path.join(Datadirectory, category)
       class_num = Classes.index(category) # 0 1,
       for img in os.listdir(path):
           try:
               img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
               backtorgb = cv2.cvtColor(img_array,cv2.COLOR_GRAY2RGB)
               new_array = cv2.resize(backtorgb, (img_size, img_size))
               training_Data.append([new_array,class_num])
           except Exception as e:
               pass
create_training_Data()

#random to avoid overfitting
random.shuffle(training_Data)


#Creating arrays to store features and labels
X = []
Y = []
for features,label in training_Data:
  X.append(features)
  Y.append(label)
X = np.array(X).reshape(-1, img_size, img_size, 3)

#CNN
model = tf.keras.applications.mobilenet.MobileNet()
base_input = model.layers[0].input ##input
base_output = model.layers[-4].output
Flat_layers = layers.Flatten()(base_output)
final_output = layers.Dense(1)(Flat_layers)
final_output = layers.Activation('sigmoid')(final_output)
new_model = keras.Model(inputs = base_input, outputs = final_output)
new_model.compile(loss="binary_crossentropy", optimizer= "adam", metrics=["accuracy"])
new_model.fit(X,Y, epochs = 5, validation_split = 0.2) ##training
new_model.save('my_model.h5')
img_array = cv2.imread('mrlEyeDataset/Closed_Eyes/s0001_00080_0_0_0_0_0_01.png', cv2.IMREAD_GRAYSCALE)
backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
new_array = cv2.resize(backtorgb, (img_size, img_size))
X_input = np.array(new_array).reshape(1, img_size, img_size, 3)
X_input = X_input/255.0 #normalizing data
prediction = new_model.predict(X_input)

