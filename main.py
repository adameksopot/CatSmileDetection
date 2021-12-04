from classes.cat_bot import CatBot
from classes.driverbuilder import DriverBuilder

driver_builder = DriverBuilder()
cat_bot = CatBot(driver=driver_builder.driver)
cat_bot.extract_data('happy')
