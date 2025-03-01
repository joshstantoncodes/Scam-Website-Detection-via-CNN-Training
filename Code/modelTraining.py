"""  Scam Website Detection Model Training


    Author: Josh Stanton
    Date: February 27, 2025
"""
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from imageTransformation import load_images, gaussian_sobel_images, uniform_sizing_via_filepath
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
import time
from tensorflow.python.keras.backend import epsilon
# PyCharm doesn't seem to like Keras but this works to import tensorflow.keras.layers
keras = tf.keras
KL = keras.layers
KW = keras.wrappers


# Loading all PNG files in a single run is extremely memory intensive and time-consuming,
scam_full_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Training Data\Scam Websites\Scam Site Captures"
scam_small_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Training Data\Scam Websites\Scam Sites Small Batch"
sobel_scam_full_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Training Data\Scam Websites\Scam Site Sobel Transformed"


# for the time being, while still building, using a smaller batch would be better.
legitimate_full_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Training Data\Legitimate Websites\Legitimate Captures"
legitimate_small_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Training Data\Legitimate Websites\Legitimate Site Small Batch"
sobel_legitimate_full_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Training Data\Legitimate Websites\Legitimate Sobel Transformed"

'''
The Convolutional Neural Network Model implements three Convolutional Building Blocks
as described in An Introduction to Image Classification by Klaus D. Toennies. Each Block
of the model consists of three batches of 64 3x3 filters followed by MaxPooling which reduces
spatial resolution. Normalization was also included to prevent overfitting. The final two layers
of the model are used to refine features and provide prediction via the sigmoid activation. 

'''
model = keras.Sequential([
    KL.Conv2D(64, (3, 3), activation='relu'),
    KL.Conv2D(64, (3, 3), activation='relu'),
    KL.Conv2D(64, (3, 3), activation='relu'),
    KL.MaxPooling2D((2, 2)),
    KL.BatchNormalization(
        momentum=0.5,
        epsilon=0.00001,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones'
    ),
    KL.Dropout(0.25),
    KL.Conv2D(64, (3, 3), activation='relu'),
    KL.Conv2D(64, (3, 3), activation='relu'),
    KL.Conv2D(64, (3, 3), activation='relu'),
    KL.MaxPooling2D((2, 2)),
    KL.BatchNormalization(
        momentum=0.5,
        epsilon=0.00001,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones'
    ),
    KL.Dropout(0.25),
    KL.Conv2D(64, (3, 3), activation='relu'),
    KL.Conv2D(64, (3, 3), activation='relu'),
    KL.Conv2D(64, (3, 3), activation='relu'),
    KL.MaxPooling2D((2, 2)),
    KL.BatchNormalization(
        momentum=0.5,
        epsilon=0.00001,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones'
    ),
    KL.Flatten(),
    KL.Dense(64, activation='relu'),
    KL.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


start_time = time.time()

# Collect all of the images in a uniform size
scam_images = uniform_sizing_via_filepath(scam_full_batch_path)
legit_images = uniform_sizing_via_filepath(legitimate_full_batch_path)


# Convert to an np.array for training split
scam_images = np.array(scam_images)
legit_images = np.array(legit_images)

# assign labels for each of the groups.
scam_labels = np.zeros(len(scam_images))
legit_labels = np.ones(len(legit_images))

# add the labels to the data
all_images = np.concatenate([scam_images, legit_images], axis=0)
all_labels = np.concatenate([scam_labels, legit_labels], axis=0)

scam_train, scam_test, legit_train, legit_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

scam_train = scam_train.astype("float32") / 255.0
scam_test = scam_test.astype("float32") / 255.0


# Trying  to run on a batch size of 32 exceeds RAM capabilities for this device.
model.fit(scam_train, legit_train, epochs=10, batch_size=8)

legit_pred = model.predict(scam_test)
legit_pred = (legit_pred >= 0.5).astype(int).flatten()

model.summary()
accuracy = accuracy_score(legit_test, legit_pred)
print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(legit_test, legit_pred))


cm = confusion_matrix(legit_test, legit_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
print('Prediction on test data:\n')
disp.plot()
plt.show()

end_time = time.time()

print(f'First Model Time Elapse: {end_time-start_time}s')

######## MODEL 2 SOBEL TRANSFORMED

start_time = time.time()
# Collect all of the images in a uniform size
scam_images = uniform_sizing_via_filepath(sobel_scam_full_batch_path)
legit_images = uniform_sizing_via_filepath(sobel_legitimate_full_batch_path)

# Convert to an np.array for training split
scam_images = np.array(scam_images)
legit_images = np.array(legit_images)

# assign labels for each of the groups.
scam_labels = np.zeros(len(scam_images))
legit_labels = np.ones(len(legit_images))

# add the labels to the data
all_images = np.concatenate([scam_images, legit_images], axis=0)
all_labels = np.concatenate([scam_labels, legit_labels], axis=0)

scam_train, scam_test, legit_train, legit_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

scam_train = scam_train.astype("float32") / 255.0
scam_test = scam_test.astype("float32") / 255.0


# Trying  to run on a batch size of 32 exceeds RAM capabilities for this device.
model.fit(scam_train, legit_train, epochs=10, batch_size=8)

legit_pred = model.predict(scam_test)
legit_pred = (legit_pred >= 0.5).astype(int).flatten()

model.summary()
accuracy = accuracy_score(legit_test, legit_pred)
print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(legit_test, legit_pred))


cm = confusion_matrix(legit_test, legit_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
print('Prediction on test data:\n')
disp.plot()
plt.show()

end_time = time.time()

print(f'Second Model Time Elapse: {end_time-start_time}s')



