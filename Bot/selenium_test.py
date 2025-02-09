# Import necessary libraries
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Driver Object created
driver = webdriver.Edge()

# Get website
# driver.get('http://mycapstone.free.nf/views/login.php')
driver.get('http://localhost/Projects/CAPSTONE/Website/views/login.php')


# Find necessary elements and input data
element_email = driver.find_element(By.ID, 'email').send_keys('john@example.com')
element_password = driver.find_element(By.ID, 'password').send_keys('Hello12345!')

# Submit form
WebDriverWait(driver, 30)
element_form = driver.find_element(By.TAG_NAME, 'form').submit()

time.sleep(30)
driver.quit()