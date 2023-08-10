# -*- coding: utf-8 -*-
"""Cancer Detection using HMM .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xbmnIajnY3giwAgGxJqTNMC2jIlMJDku
"""

!pip install pydicom



import cv2
import numpy as np
import pydicom as PDCM



# from google.colab import drive
# drive.mount('/content/drive')

dataset = PDCM.read_file('/content/drive/MyDrive/CT_dataset/dicom_dir/ID_0000_AGE_0060_CONTRAST_1_CT.dcm')

print(dataset.pixel_array)

dataset.pixel_array.shape

import matplotlib.pyplot as plt
plt.imshow(dataset.pixel_array , cmap='gray')
plt.show()

dataset

path = '/content/drive/MyDrive/CT_dataset/dicom_dir/'

import os
import cv2
import numpy as np
import pydicom as dicom

images = []
for file in os.listdir(path):
    f = dicom.read_file(os.path.join(path,file)).pixel_array
    fr = cv2.resize(f , (256,256))
    images.append(fr)

X = np.array(images)
X = X.reshape(100,256,256,1)
X.shape

labels = []
for file in os.listdir(path):
    labels.append(file.split('_')[-2])

Y = np.array(labels)
Y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)

x_train.shape

import tensorflow as tf

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

from keras import layers, models, regularizers

"""**Neural Network **

Traditional CNN
"""

model = models.Sequential()

model.add(layers.Conv2D(64 ,(3,3) , padding = 'same' , kernel_regularizer=regularizers.l2(0.0001) , input_shape = (256,256,1)))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(64 ,(3,3) , kernel_regularizer=regularizers.l2(0.0001) , padding = 'same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(128 ,(3,3) , kernel_regularizer=regularizers.l2(0.0001) , padding = 'same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(128 ,(3,3) , kernel_regularizer=regularizers.l2(0.0001) , padding = 'same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(256 ,(3,3) , kernel_regularizer=regularizers.l2(0.0001) , padding = 'same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(256 ,(3,3) , kernel_regularizer=regularizers.l2(0.0001) , padding = 'same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(256 ,(3,3) , kernel_regularizer=regularizers.l2(0.0001) , padding = 'same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(512 ,(3,3) , kernel_regularizer=regularizers.l2(0.0001) , padding = 'same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(512 ,(3,3) , kernel_regularizer=regularizers.l2(0.0001) , padding = 'same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(512 ,(3,3) , kernel_regularizer=regularizers.l2(0.0001) , padding = 'same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(512 ,(3,3) , kernel_regularizer=regularizers.l2(0.0001) , padding = 'same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(512 ,(3,3) , kernel_regularizer=regularizers.l2(0.0001) , padding = 'same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(512 ,(3,3) , kernel_regularizer=regularizers.l2(0.0001) , padding = 'same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))


model.add(layers.Conv2D(4096 ,(3,3) , kernel_regularizer=regularizers.l2(0.0001) , padding = 'same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(2, activation='sigmoid') )

model.summary()

import tensorflow as tf

sgd = tf.keras.optimizers.SGD(0.001)
model.compile(loss='categorical_crossentropy' , optimizer=sgd , metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range = 40,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range = 0.2,
                             zoom_range = 0.2,
                             horizontal_flip = True,
                             brightness_range = (0.5, 1.5))

history = model.fit(datagen.flow(x_train,y_train,batch_size=10),
         epochs=20,
         validation_data=datagen.flow(x_test,y_test))

import matplotlib.pyplot as plt

# Extracted metrics from the training history
num_epochs =20
epochs = range(1, num_epochs + 1)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

# Plot training and validation accuracy
plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save("model.h5")

from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score

# Make predictions on the test data
y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class labels

# Convert one-hot encoded labels to integer labels
y_true = np.argmax(y_test, axis=1)

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Print the results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("F1-score:", f1)