"""  Scam Website Detection Model Training


    Author: Josh Stanton
    Date: February 27, 2025
"""
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import KFold, train_test_split
from imageTransformation import load_images, gaussian_sobel_images
import cv2 as cv
import sklearn
import math
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import epsilon
# PyCharm doesn't seem to like Keras but this works to import tensorflow.keras.layers
keras = tf.keras
KL = keras.layers


# Loading all PNG files in a single run is extremely memory intensive and time-consuming,
scam_full_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Scam Websites\Scam Site Captures"
scam_small_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Scam Websites\Scam Sites Small Batch"

# for the time being, while still building, using a smaller batch would be better.
legitimate_full_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Legitimate Websites\Legitimate Captures"
legitimate_small_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Legitimate Websites\Legitimate Site Small Batch"


'''
The Convolutional Neural Network Model implements three Convolutional Building Blocks
as described in An Introduction to Image Classification by Klaus D. Toennies. Each Block
of the model consists of three batches of 64 3x3 filters followed by MaxPooling which reduces
spatial resolution. Normalization was also included to prevent overfitting. The final two layers
of the model are used to refine features and provide prediction via the softmax activation. 

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
    #KL.Dropout(0.25),
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
    #KL.Dropout(0.25),
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
    KL.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

scam_images = load_images(scam_small_batch_path)


legit_images = load_images(legitimate_small_batch_path)


# Let's create two test groups, the raw OpenCV pixel matrices and the Sobel Operator transformed pixel matrices.
scam_untransformed = sorted(scam_images, key=len, reverse=True)
legit_untransformed = sorted(legit_images, key=len, reverse=True)




print(f'Length of the first webpage in the scams: {len(scam_untransformed[0])}, last webpage {len(scam_untransformed[len(scam_untransformed)-1])}')
print(f'Length of the first webpage in the legit: {len(legit_untransformed[0])}, last webpage {len(legit_untransformed[len(legit_untransformed)-1])}')


# scam_train, scam_test, legit_train, legit_test = train_test_split(scam_untransformed, legit_untransformed, random_state=104, test_size=0.2, shuffle=True)
'''
kf = KFold(n_splits=5, shuffle=True, random_state=104)
for train_index, test_index in kf.split(X_untransformed):
    scam_train, scam_test = scam_untransformed.iloc[train_index], scam_untransformed.iloc[test_index]
    legit_train, legit_test = legit_untransformed.iloc[train_index], legit_untransformed.iloc[test_index]
'''



'''
model.fit(scam_train, legit_train, epochs=25)
legit_pred = model.predict(scam_test)

accuracy = accuracy_score(legit_test, legit_pred)
print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(legit_test, legit_pred))


cm = confusion_matrix(legit_test, legit_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
print('Prediction on test data:\n')
disp.plot()
plt.show()

'''






