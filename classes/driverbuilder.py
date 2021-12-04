import logging
import os

from selenium.common.exceptions import SessionNotCreatedException, WebDriverException
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


class DriverBuilder:
    def __init__(self, webkit_driver_path=None, headless_mode=False):
        options = Options()
        options.headless = headless_mode
        options.add_argument("--lang=en")
        options.add_argument("log-level=3")

        if webkit_driver_path is None:
            webkit_driver_path = os.path.join(os.getcwd(), "chromedriver.exe")

        try:
            self.driver = webdriver.Chrome(
                chrome_options=options,
                executable_path=webkit_driver_path,
                service_log_path='NUL',
            )

        except (SessionNotCreatedException, WebDriverException) as e:
            logging.info(e)
            self.driver = webdriver.Chrome(
                chrome_options=options,
                executable_path=webkit_driver_path,
                service_log_path='NUL',

            )
