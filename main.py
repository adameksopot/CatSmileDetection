import logging
import os
from tkinter import Tk
from classes.cat_bot import CatBot
from classes.driverbuilder import DriverBuilder

# driver_builder = DriverBuilder()
# cat_bot = CatBot(driver=driver_builder.driver)
# cat_bot.extract_data('happy')
from classes.gui import MyFirstGUI


root = MyFirstGUI()
root.mainloop()
import Augmentor


def augmentation(folder):  # "out_smiling_cut"
    p = Augmentor.Pipeline(folder)
    p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
    p.random_brightness(
        probability=.5,
        min_factor=.5,
        max_factor=1.4)
    p.flip_left_right(0.5)
    p.sample(500)
    p.set_save_format(save_format="auto")
