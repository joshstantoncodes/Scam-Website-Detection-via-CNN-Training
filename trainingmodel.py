"""  Scam Website Detection Site Capture
    To train the CNN, the images must be able to be analyzed by the program,
    and images will be read into the program using the OpenCV package.

    Author: Josh Stanton
    Date: February 16, 2025
"""

import pandas as pd
import numpy as np
import os
import cv2 # imports the package for OpenCV's Library

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
      An array of read images
      """
    images = []
    # collect all PNG files in the directory
    img_files = [files for files in os.listdir(filepath) if files.endswith('.png')]

    for file in img_files:
        full_path = os.path.join(filepath, file)
        img = cv2.imread(full_path) # OpenCV struggles with reading nondirect filepaths
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

print(images[0])


'''
scam_site = cv2.imread(image_path)

cv2.imshow("scam_site", scam_site)

cv2.waitKey(0)

cv2.destroyAllWindows()
'''