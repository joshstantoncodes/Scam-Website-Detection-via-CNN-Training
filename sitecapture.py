"""  Scam Website Detection Site Capture
    In order to train a CNN based on image recognition of fraudulent and scam websites,
    images of the entire site must be gathered in full so that they can be used for training.
    Manually going through every single website would take an inordinate amount of time, thus
    programming Selenium to take the screenshots is the optimal path.

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
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile # Used to remove securities, only use when contained
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
    ''' ONLY USE WITHIN A VM FOR SAFETY
    
    profile = FirefoxProfile()
    profile.set_preference("security.insecure_field_warning.contextual.enabled", False) # [6]
    profile.set_preference("browser.safebrowsing.malware.enabled", False)
    profile.set_preference("browser.safebrowsing.phishing.enabled", False)
    profile.set_preference("security.ssl.enable_ocsp_stapling", False)
    profile.set_preference("security.OCSP.enabled", 0)
    profile.set_preference("security.ssl.errorReporting.automatic", False)
    profile.set_preference("security.ssl.errorReporting.enabled", False)
    profile.set_preference("security.ssl.errorReporting.url", "")
    profile.set_preference("security.ssl.require_safe_negotiation", False)
    profile.set_preference("security.mixed_content.block_active_content", False)
    profile.set_preference("security.mixed_content.block_display_content", False)
    profile.set_preference("network.stricttransportsecurity.preloadlist", False)
    profile.set_preference("network.stricttransportsecurity.enabled", False)
    profile.set_preference("network.http.phishy-userpass-length", 255)
    profile.set_accept_untrusted_certs(True)
    profile.set_assume_untrusted_certificate_issuer(False)
    
    '''
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=options)
    # launches the website URL in the driver
    driver.get(url)
    # sets an implicit wait time of 30 seconds to allow for the webpage to load fully
    driver.implicitly_wait(30)
    # set the page sizing using the JavaScript command to capture the full webpage
    full_page = driver.execute_script("return document.body.scrollHeight")
    driver.set_window_size(1920, full_page)
    driver.save_full_page_screenshot(
        os.path.join(r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Scam Site Captures",
                                                  f"screenshot_{url_index}.png"))
    driver.quit()


for index, row in scam_sites.iterrows():
    labels = ['URL']
    for label in labels:
        acquire_screenshot(row[label], index)

