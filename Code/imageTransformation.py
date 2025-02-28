"""  Scam Website Detection Image Preprocessing
    To train the CNN, the images must be able to be analyzed by the program,
    and images will be read into the program using the OpenCV package.

    ~ Could investigate if results outline that scam websites prefer a specific
    Javascript visual framework over others **

    Author: Josh Stanton
    Date: February 16, 2025
"""

import os
import cv2 as cv  # imports the package for OpenCV's Library
import numpy as np


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
        img = cv.imread(full_path)  # OpenCV struggles with reading nondirect filepaths
        if img is not None:
            images.append(img)
        else:
            print(f'Error loading image: {file}')

    if images:
        print(f'Successfully loaded {len(images)} PNG files')
    else:
        print('Images not loaded')

    return images


# Now to modify the landing pages for easier feature detection

def gaussian_sobel_images(filepath_in: str, filepath_out: str):
    """
      Convenience function for loading the PNG files and transforming them with Gaussian Blur and Sobel
      operation for edge detection

      Parameters
      ----------
      filepath_in : str
          The name of the directory (absolute or relative) containing images to be transformed.

      filepath_out : str
          The name of the directory (absolute or relative) where the transformed images will be saved

      Returns
      -------
      A gaussian blurred set of images.

      """
    images = []
    # collect all PNG files in the directory
    img_files = [files for files in os.listdir(filepath_in) if files.endswith('.png')]

    for file in img_files:
        full_path = os.path.join(filepath_in, file)
        img = cv.imread(full_path)
        # img = cv.GaussianBlur(img,(3,3), 0)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # The Sobel operator is used to compute an approximation of the gradient, combining the Gaussian smoothing
        # and differentiation, allowing for sharper edge detection.
        # ksize = kernel size, high kernel, higher smoothing
        x_grad = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        y_grad = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        abs_grad_x = cv.convertScaleAbs(x_grad)
        abs_grad_y = cv.convertScaleAbs(y_grad)
        img = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        images.append(img)
        os.chdir(filepath_out)
        cv.imwrite(file, img)
    return images

def uniform_sizing_via_filepath(filepath: str):
    """
      In order to train the model, the captures must all be of the same size and shape, so let's
      transform these images into a uniform size. Some research has indicated that one of the most
      common dimensions for websites. On top of that, padding is applied in the event that uniformity
      is a challenge.

      Parameters
      ----------
      filepath : str
          The name of the directory (absolute or relative) containing data.

      Returns
      -------
      Uniformly resized images.
      """
    # let's try the most common website dimensions at the moment
    desired_height, desired_width = 960, 540
    images = []
    # collect all PNG files in the directory
    img_files = [files for files in os.listdir(filepath) if files.endswith('.png')]

    for file in img_files:
        full_path = os.path.join(filepath, file)
        img = cv.imread(full_path)
        height, width = img.shape[:2]
        scale = min(desired_height / height, desired_width / width)

        new_height = int(height * scale)
        new_width = int(width * scale)
        resized_image = cv.resize(img, (new_width, new_height))
        pad_top = (desired_height - new_height) // 2
        pad_bottom = desired_height - new_height - pad_top
        pad_left = (desired_width - new_width) // 2
        pad_right = desired_width - new_width - pad_left
        padded_resized_image = cv.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right, cv.BORDER_CONSTANT)
        images.append(padded_resized_image)
    return images


# Loading all PNG files in a single run is extremely memory intensive and time-consuming,
scam_full_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Scam Websites\Scam Site Captures"
scam_small_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Scam Websites\Scam Sites Small Batch"

# for the time being, while still building, using a smaller batch would be better.
legitimate_full_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Legitimate Websites\Legitimate Captures"
legitimate_small_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Legitimate Websites\Legitimate Site Small Batch"
legit_sobel_out = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Legitimate Websites\Legitimate Sobel Transformed"
scam_sobel_out = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Scam Websites\Scam Site Sobel Transformed"


# Once run, not needed to run again for this set of work.
# gaussian_sobel_images(scam_full_batch_path, scam_sobel_out)
# gaussian_sobel_images(legitimate_full_batch_path,legit_sobel_out)




