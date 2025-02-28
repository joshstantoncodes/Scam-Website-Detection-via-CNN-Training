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

scam_images = uniform_sizing_via_filepath(scam_small_batch_path)
legit_images = uniform_sizing_via_filepath(legitimate_small_batch_path)

print(type(scam_images), type(legit_images))

'''

Layer "conv2d" expects 1 input(s), but it received 80 input tensors.
Inputs received: [<tf.Tensor 'data:0' shape=(32, 1080, 3) dtype=uint8>, 
<tf.Tensor 'data_1:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_2:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 
'data_3:0' shape=(32, 1080, 3)
 dtype=uint8>, <tf.Tensor 'data_4:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_5:0' shape=(32, 1080, 3) dtype=uint8>,
  <tf.Tensor 'data_6:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_7:0' shape=(32, 1080, 3) dtype=uint8>, 
  <tf.Tensor 'data_8:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_9:0' shape=(32, 1080, 3) dtype=uint8>, 
  <tf.Tensor 'data_10:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_11:0' shape=(32, 1080, 3) dtype=uint8>,
   <tf.Tensor 'data_12:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_13:0' shape=(32, 1080, 3) dtype=uint8>, 
   <tf.Tensor 'data_14:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_15:0' shape=(32, 1080, 3) dtype=uint8>, 
   <tf.Tensor 'data_16:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_17:0' shape=(32, 1080, 3) dtype=uint8>, 
   <tf.Tensor 'data_18:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_19:0' shape=(32, 1080, 3) dtype=uint8>, 
   <tf.Tensor 'data_20:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_21:0' shape=(32, 1080, 3) dtype=uint8>, 
   <tf.Tensor 'data_22:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_23:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_24:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_25:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_26:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_27:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_28:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_29:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_30:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_31:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_32:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_33:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_34:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_35:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_36:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_37:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_38:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_39:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_40:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_41:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_42:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_43:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_44:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_45:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_46:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_47:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_48:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_49:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_50:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_51:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_52:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_53:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_54:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_55:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_56:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_57:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_58:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_59:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_60:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_61:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_62:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_63:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_64:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_65:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_66:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_67:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_68:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_69:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_70:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_71:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_72:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_73:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_74:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_75:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_76:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_77:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_78:0' shape=(32, 1080, 3) dtype=uint8>, <tf.Tensor 'data_79:0' shape=(32, 1080, 3) dtype=uint8>]

Arguments received by Sequential.call():
  • inputs=('tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)', 'tf.Tensor(shape=(32, 1080, 3), dtype=uint8)')
  • training=True
  • mask=('None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 
  'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 
  'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None')


'''



scam_train, scam_test, legit_train, legit_test = train_test_split(scam_images, legit_images, random_state=104, test_size=0.2, shuffle=True)
'''
kf = KFold(n_splits=5, shuffle=True, random_state=104)
for train_index, test_index in kf.split(X_untransformed):
    scam_train, scam_test = scam_images.iloc[train_index], scam_images.iloc[test_index]
    legit_train, legit_test = legit_images.iloc[train_index], legit_images.iloc[test_index]
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







