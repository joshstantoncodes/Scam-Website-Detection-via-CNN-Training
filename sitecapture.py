"""  Scam Website Detection Site Capture


    Author: Josh Stanton
    Date: February 06, 2025
"""

import pandas as pd
#import numpy as np
import os
import glob # used for UNIX style path-name
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager


# Using the absolute file path of the Scam Websites Folder, collects all the .csv files into a single DataFrame
file_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Scam Websites"

def load_data(filepath: str) -> pd.DataFrame:
    """
      Convenience function for loading the given .csv files in the specified filepath into a
      single DataFrame.

      Parameters
      ----------
      filepath : str
          The name of the directory (absolute or relative) containing data.

      Returns
      -------
      A single DataFrame containing all data from the directory.
      """
    data_files = glob.glob(os.path.join(filepath, "*.csv"))
    files = []
    for file_name in data_files:
        df = pd.read_csv(file_name, index_col=0, header=0)
        files.append(df)
    data = pd.concat(files, axis=0, ignore_index=True)
    return data

scam_sites = load_data(file_path)

def acquire_screenshot(url: str, url_index: int):
    """
      Given a website's URL, generates a selenium Firefox driver and captures a screenshot
      of the webpage and saves the screenshot, exiting the program after completing the screenshot

      To run without requiring the driver producing a GUI instance of Firefox, set
      options to headless, otherwise remove.

      Parameters
      ----------
      url : str
          The URL of a website.
      url_index : int
      An indexing number for the screenshot, corresponding to the URL element in the DataFrame

      Returns
      -------
      A .png file of the screenshot taken.
      """
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=options)
    # launches the website URL in the driver
    driver.get(url)
    # sets an implicit wait time of 15 seconds to allow for the webpage to load fully
    driver.implicitly_wait(15)
    # set the page sizing using the JavaScript command to capture the full webpage
    full_page = driver.execute_script("return document.body.scrollHeight")
    driver.set_window_size(1920, full_page)
    driver.save_full_page_screenshot(
        os.path.join(r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Scam Site Captures",
                                                  f"screenshot_{url_index}.png"))
    driver.quit()


acquire_screenshot("https://urlbox.com/website-screenshots-python",1)

'''
for index, row in scam_sites.iterrows():
    labels = ['URL']
    for label in labels:
        acquire_screenshot(row[label], index)

'''