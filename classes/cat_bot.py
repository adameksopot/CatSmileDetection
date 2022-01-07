import logging
from selenium.webdriver.support import ui
import requests


class CatBot:
    """
        class for downloading picutres of smiling and grumpy cats
    """
    def __init__(self, driver):
        self.driver = driver
        self.timeout = 60 * 3
        self.wait = ui.WebDriverWait(self.driver, self.timeout)

    def extract_data(self, emotion):
        try:
            url = "https://www.flickr.com/search/?text={0}%20cat".format(emotion)
            self.driver.get(url)
            images = self.driver.find_elements_by_class_name("overlay")

            for i, image in enumerate(images):
                image_url = images[i].get_attribute('href')
                self.driver.execute_script("window.open(arguments[0], 'new_window')", image_url)

                self.driver.switch_to.window(self.driver.window_handles[1])
                img = self.driver.find_element_by_class_name("main-photo")
                url = img.get_attribute('src')
                img_data = requests.get(url).content
                with open('out/cat_{}_{}.jpg'.format(emotion, i), 'wb') as handler:
                    handler.write(img_data)
                print('cat_{}_{} downloaded'.format(emotion, i))
                self.driver.close()
                self.driver.switch_to.window(self.driver.window_handles[0])
        except Exception as e:
            logging.info(e)
