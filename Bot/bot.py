# Bot program goes here
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Specify the path to ChromeDriver
driver_path = "/path/to/chromedriver"
driver = webdriver.Chrome(driver_path)

# URL of the login page
login_url = "localhost/Projects/CAPSTONE/Website/views/login.php"
driver.get(login_url)

# Give time for the page to load
time.sleep(2)

# Locate the username and password fields
email_field = driver.find_element(By.NAME, "email")  # Adjust based on the element's attribute
password_field = driver.find_element(By.NAME, "password")  # Adjust based on the element's attribute

# Test: Print out to confirm the elements are found
print("Email field:", email_field)
print("Password field:", password_field)

# Close the browser after the test
time.sleep(2)
driver.quit()

#----------------------------------------------------------------------------
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options

# Set up Edge options
options = Options()
options.use_chromium = True

# Set up Edge service
service = Service('path/to/msedgedriver')

# Create a new Edge session
driver = webdriver.Edge(service=service, options=options)

try:
# Navigate to Bing
driver.get("https://bing.com")

# Find the search box and enter a query
search_box = driver.find_element_by_id("sb_form_q")
search_box.send_keys("WebDriver")
search_box.submit()

# Wait for a few seconds to see the results
driver.implicitly_wait(5)
finally:
# Close the browser
driver.quit()
