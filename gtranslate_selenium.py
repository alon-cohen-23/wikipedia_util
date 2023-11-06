import os
import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pathlib import Path
import time
from plyer import notification
from loguru import logger

# Define button constants
FILE_INPUT_NAME = "file"
DOCUMENTS_BUTTON_XPATH = "//button[span[contains(., 'Documents')]]"
TRANSLATE_BUTTON_XPATH = "//button[span[contains(., 'Translate')]]"
DOWNLOAD_BUTTON_XPATH = "//button[span[contains(., 'Download translation')]]"
CLEAR_FILE_BUTTON_CSS_SELECTOR = "button[aria-label='Clear file']"


def google_translate_folder_of_excels(folder_path, chromedriver_path):
    """

    Parameters
    ----------
    folder_path : Path to folder with .xlsx files to translate
    chromedriver_path: Path to the chromr driver 
    ***** Important: <= 1MB per file, as it may refuse to translate several larger files in seq 
    -------
    None.

    """
    options =  webdriver.ChromeOptions() #uc.options.ChromeOptions()

    # Set the language to English
    options.add_argument("--lang=en")
    options.headless = False
    options.add_argument(f'--chromedriver-executable={chromedriver_path}')

    # Create a new Chrome driver instance
    browser =  webdriver.Chrome(options=options) #uc.Chrome(options)
    
    
    # Get the Windows Downloads folder path
    downloads_folder = os.path.expanduser("~/Downloads")
    
    # Loop through all .xlsx files in the folder and upload them one by one
    for file_path in Path(folder_path).glob("*.xlsx"):
        try:
            # Check if the file already exists in the Downloads folder, and if so, skip it
            if os.path.exists(os.path.join(downloads_folder, file_path.name)):
                logger.info(f"Skipping {file_path.name} as it already exists in the Downloads folder.")
                continue
            
            # Navigate to the desired website
            browser.get("https://translate.google.com/?hl=iw&sl=auto&tl=en&op=docs")
            
            # Use WebDriverWait to wait for the "Translate" button with a span containing "Translate"
            translate_button = WebDriverWait(browser, 120).until(
                EC.element_to_be_clickable((By.XPATH, DOCUMENTS_BUTTON_XPATH))
            )
            
            # Log the file currently being translated
            logger.info(f"Translating {file_path.name}")
            
            file_input = WebDriverWait(browser, 30).until(
                EC.presence_of_element_located((By.NAME, FILE_INPUT_NAME))
            )
            
            # Log info before uploading the file
            logger.info(f"Uploading {file_path.name}")
            file_input.send_keys(str(file_path))
            
            # Use WebDriverWait to wait for the "Translate" button with a span containing "Translate"
            translate_button = WebDriverWait(browser, 120).until(
                EC.element_to_be_clickable((By.XPATH, TRANSLATE_BUTTON_XPATH))
            )
            
            # Log info before clicking the "Translate" button
            logger.info("Clicking the 'Translate' button")
            translate_button.click()
                        
            # Now that it's "Translating...", wait to click the "Download" button
            download_button = WebDriverWait(browser, 120).until(
                EC.element_to_be_clickable((By.XPATH, DOWNLOAD_BUTTON_XPATH))
            )
            
            # Log info before clicking the "Download" button
            logger.info("Clicking the 'Download' button")
            download_button.click()
            
            
            # Log success message
            logger.info(f"Translation of {file_path.name} completed successfully.")
        
        except Exception as e:
            # Log error and continue to the next file
            logger.error(f"Error while translating {file_path.name}: {str(e)}")
            continue
        
    time.sleep(120) # wait for last file to download (or result is a partial download file)    
    browser.close()
    browser.quit()

# Call the function with the desired folder path
folder_path = r"H:/Alon/wikipedia_util/tr_dataset"
chromedriver_path = 'H:/Alon/wikipedia_util/chromedriver.exe'
google_translate_folder_of_excels(folder_path, chromedriver_path)
