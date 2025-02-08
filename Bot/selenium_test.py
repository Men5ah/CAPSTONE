# Import necessary libraries
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Driver Object created
driver = webdriver.Edge()

# Get website
driver.get('http://mycapstone.free.nf/Website/views/login.php?i=1')


# Find necessary elements and input data
element_email = driver.find_element(By.ID, 'email').send_keys('john@example.com')
element_password = driver.find_element(By.ID, 'password').send_keys('Hello12345!')


# Switch context for captcha
WebDriverWait(driver, 10).until(
    EC.frame_to_be_available_and_switch_to_it((By.XPATH, "//iframe[contains(@src, 'recaptcha/api2/anchor')]"))
)
WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.CSS_SELECTOR, 'div.recaptcha-checkbox-border'))
).click()
driver.switch_to.default_content()

# Submit form
element_form = driver.find_element(By.TAG_NAME, 'form').submit()

time.sleep(10)
driver.quit()