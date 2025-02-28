"""  Scam Website Detection Model Training


    Author: Josh Stanton
    Date: February 27, 2025
"""
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import KFold, train_test_split
from tensorflow.python.ops.gen_uniform_quant_ops import uniform_quantized_clip_by_value

from imageTransformation import load_images, gaussian_sobel_images, uniform_sizing_via_filepath
import cv2 as cv
import sklearn
import math
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
import time
from tensorflow.python.keras.backend import epsilon
# PyCharm doesn't seem to like Keras but this works to import tensorflow.keras.layers
keras = tf.keras
KL = keras.layers


# Loading all PNG files in a single run is extremely memory intensive and time-consuming,
scam_full_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Training Data\Scam Websites\Scam Site Captures"
scam_small_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Training Data\Scam Websites\Scam Sites Small Batch"

# for the time being, while still building, using a smaller batch would be better.
legitimate_full_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Training Data\Legitimate Websites\Legitimate Captures"
legitimate_small_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Training Data\Legitimate Websites\Legitimate Site Small Batch"


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

start_time = time.time()

scam_images = uniform_sizing_via_filepath(scam_small_batch_path)
legit_images = uniform_sizing_via_filepath(legitimate_small_batch_path)

scam_images = np.array(scam_images)
legit_images = np.array(legit_images)

scam_images = tf.data.Dataset.from_tensor_slices(scam_images)
legit_images = tf.data.Dataset.from_tensor_slices(legit_images)

scam_images = scam_images.batch(64)
legit_images = legit_images.batch(64)


scam_train, scam_test, legit_train, legit_test = train_test_split(scam_images, legit_images, random_state=104, test_size=0.2, shuffle=True)
'''
kf = KFold(n_splits=5, shuffle=True, random_state=104)
for train_index, test_index in kf.split(X_untransformed):
    scam_train, scam_test = scam_images.iloc[train_index], scam_images.iloc[test_index]
    legit_train, legit_test = legit_images.iloc[train_index], legit_images.iloc[test_index]
'''


# reshape the tensors to a single tensor for the model to fit properly.
'''
scam_train = tf.reshape(scam_train, [-1])
scam_test = tf.reshape(scam_test, [-1])
legit_train = tf.reshape(legit_train, [-1])
legit_test = tf.reshape(legit_test, [-1])
'''
scam_train = tf.uniform(32, 1080, 1920, 3)

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

end_time = time.time()


print(f'Time Elapse: {end_time-start_time}s')





