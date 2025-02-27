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
import requests
from requests.auth import HTTPBasicAuth
from urllib3.exceptions import NameResolutionError, MaxRetryError, HTTPError


# Using the absolute file path of the Scam Websites Folder, collects all the .csv files into a single DataFrame
file_path = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Scam Websites"
legitimate = r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Legitimate Websites"

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
legit_sites = load_data(legitimate)

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

    '''
    profile = FirefoxProfile()
    profile.set_preference("security.insecure_field_warning.contextual.enabled", False)
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
    '''

    options = Options()
    options.add_argument("--headless")
    service = FirefoxService(GeckoDriverManager().install())
    driver = webdriver.Firefox(service=service, options=options)
    # launches the website URL in the driver
    try:
        driver.get(url)
        # sets an implicit wait time of 30 seconds to allow for the webpage to load fully
        driver.implicitly_wait(30)
        # set the page sizing using the JavaScript command to capture the full webpage
        full_page = driver.execute_script("return document.body.scrollHeight")
        driver.set_window_size(1920, full_page)
        driver.save_full_page_screenshot(
        os.path.join(r"C:\Users\joshs\Documents\GitHub\Scam-Website-Detection-via-CNN-Training\Legitimate Captures",
                                                  f"screenshot_{url_index}.png"))
    except Exception as e:
        print(f"Error capturing {url} as website may have already been taken down or the domain has changed: {e}")
    finally:
        driver.quit()


def evaluate_URL(URL: str) -> bool:
    """
      To prevent issues arising during the screenshot acquisition, first assess if the URL is still active
      by sending an HTTP request, and to prevent reaching HTTP Request limits, set a request limit first.

      Parameters
      ----------
      URL : str
          The full URL of a website to send an HTTP request to.

      Returns
      -------
      True if the HTTP request was successfully received, understood, and accepted.
      False, otherwise
      """
    try:
        req = requests.Request('GET', URL)
        r = req.prepare()
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        session.mount('http://', adapter, )
        session.mount('https://', adapter)
        response = session.send(r)
        status = response.status_code

        if status == 200:
            return True

    except NameResolutionError as e:
        print(f'DNS resolution failed: {e}')
        return False

    except MaxRetryError as e:
        print(f'Max retries exceeded: {e}')
        return False

    except ConnectionError as e:
        print(f'Connection error: {e}')
        return False

    except HTTPError as e:
        print(f'HTTPError: {e}')

    except Exception as e:
        print(f'An error occurred: {e}')
        return False


'''
for index, row in scam_sites.iterrows():
    labels = ['URL']
    for label in labels:
        if evaluate_URL(row[label]):
            print(f'{index}) Acquiring screenshot from {row[label]}')
            acquire_screenshot(row[label], index)
            print(f'{index} done.')
        else:
            continue
'''
for index, row in legit_sites.iterrows():
    labels = ['URL']
    for label in labels:
        if index > 867:
            if evaluate_URL(row[label]):
                print(f'{index}) Acquiring screenshot from {row[label]}')
                acquire_screenshot(row[label], index)
                print(f'{index} done.')
            else:
                continue

