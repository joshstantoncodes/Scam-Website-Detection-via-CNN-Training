"""  Scam Website Detection Site Capture
    To train the CNN, the images must be able to be analyzed by the program,
    and images will be read into the program using the OpenCV package.

    ~ Could investigate if results outline that scam websites prefer a specific
    Javascript visual framework over others **

    Author: Josh Stanton
    Date: February 16, 2025
"""

import pandas as pd
import numpy as np
import os
import cv2 as cv# imports the package for OpenCV's Library
import numpy.matrixlib

# Loading all PNG files in a single run is extremely memory intensive and time-consuming,
# for the time being, while still building, using a smaller batch would be better.
full_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Scam Site Captures"
small_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Scam Sites Small Batch"

def load_images(filepath: str):
    """
      Convenience function for loading the PNG files into a single entity for batch analysis.
      Reading utilizes OpenCV, which uses a BGR (blue-green-red) order when processing images


      Parameters
      ----------
      filepath : str
          The name of the directory (absolute or relative) containing data.

      Returns
      -------
      A large two-dimensional matrix of all the pixel triplets of the given images.
      """
    images = []
    # collect all PNG files in the directory
    img_files = [files for files in os.listdir(filepath) if files.endswith('.png')]

    for file in img_files:
        full_path = os.path.join(filepath, file)
        img = cv.imread(full_path) # OpenCV struggles with reading nondirect filepaths
        if img is not None:
            images.append(img)
        else:
            print(f'Error loading image: {file}')

    if images:
        print(f'Successfully loaded {len(images)} PNG files')
    else:
        print('Images not loaded')

    return images

images = load_images(small_batch_path)

# Now to modify the landing pages for easier feature detection

def gaussian_sobel_images(filepath: str):
    """
      Convenience function for loading the PNG files and transforming them with Gaussian Blur and Sobel
      operation for edge detection

      Parameters
      ----------
      filepath : str
          The name of the directory (absolute or relative) containing data.

      Returns
      -------
      A gaussian blurred set of images.
      """
    images = []
    # collect all PNG files in the directory
    img_files = [files for files in os.listdir(filepath) if files.endswith('.png')]

    for file in img_files:
        full_path = os.path.join(filepath, file)
        img = cv.imread(full_path)
        #img = cv.GaussianBlur(img,(3,3), 0)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # The Sobel operator is used to compute an approximation of the gradient, combining the Gaussian smoothing
        # and differentiation, allowing for sharper edge detection.
        # ksize = kernel size, high kernel, higher smoothing
        x_grad = cv.Sobel(img, cv.CV_64F, 1,0, ksize=3, scale=1,delta=0,borderType=cv.BORDER_DEFAULT)
        y_grad = cv.Sobel(img, cv.CV_64F, 0,1, ksize=3, scale=1,delta=0,borderType=cv.BORDER_DEFAULT)
        abs_grad_x = cv.convertScaleAbs(x_grad)
        abs_grad_y = cv.convertScaleAbs(y_grad)
        img = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        images.append(img)
    return images

blur = gaussian_sobel_images(small_batch_path)

print(blur[0])


#cv.imshow("window name", blur[0])
#cv.waitKey(0)
#cv.destroyAllWindows()








