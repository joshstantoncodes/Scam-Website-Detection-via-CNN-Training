"""  Scam Website Detection Model Training


    Author: Josh Stanton
    Date: February 27, 2025
"""

from imageTransformation import load_images, gaussian_sobel_images
import cv2 as cv
import sklearn




# Loading all PNG files in a single run is extremely memory intensive and time-consuming,
# for the time being, while still building, using a smaller batch would be better.
scam_full_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Scam Websites\Scam Site Captures"
scam_small_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Scam Websites\Scam Sites Small Batch"

legitimate_full_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Legitimate Websites\Legitimate Captures"
legitimate_small_batch_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Legitimate Websites\Legitimate Site Small Batch"


scam_images = load_images(scam_small_batch_path)
scam_blur = gaussian_sobel_images(scam_small_batch_path)

legit_images = load_images(legitimate_small_batch_path)
legit_blur = gaussian_sobel_images(legitimate_small_batch_path)



cv.imshow("window name", legit_blur[0])
cv.waitKey(0)
cv.destroyAllWindows()











