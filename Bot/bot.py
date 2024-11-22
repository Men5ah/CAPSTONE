# Bot program goes here
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Specify the path to ChromeDriver
driver_path = "/path/to/chromedriver"
driver = webdriver.Chrome(driver_path)

# URL of the login page
login_url = "https://example.com/login"
driver.get(login_url)

# Give time for the page to load
time.sleep(2)

# Locate the username and password fields
username_field = driver.find_element(By.NAME, "username")  # Adjust based on the element's attribute
password_field = driver.find_element(By.NAME, "password")  # Adjust based on the element's attribute

# Test: Print out to confirm the elements are found
print("Username field:", username_field)
print("Password field:", password_field)

# Close the browser after the test
time.sleep(2)
driver.quit()
