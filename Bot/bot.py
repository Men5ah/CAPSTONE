# # # Import necessary libraries
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.action_chains import ActionChains
# import time
# import csv
# import random

# # Load credentials from CSV file
# def load_credentials(file_path):
#     credentials = []
#     with open(file_path, mode='r', encoding='utf-8') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             credentials.append({"email": row["email"], "password": row["password"]})
#     random.shuffle(credentials)  # Shuffle for randomness
#     return credentials

# # Initialize WebDriver (Edge in this case)
# def init_driver():
#     options = webdriver.EdgeOptions()
#     # options.add_argument("--headless")  # Run in headless mode (optional)
#     options.add_argument("--disable-blink-features=AutomationControlled")  # Avoid detection
#     driver = webdriver.Edge(options=options)
#     return driver

# # Move mouse to element and click
# def move_and_click(driver, element):
#     action = ActionChains(driver)
#     action.move_to_element(element).pause(random.uniform(0.3, 1)).click().perform()

# # Perform login with given credentials
# def login(driver, email, password):
#     driver.get('http://localhost/Projects/CAPSTONE/Website/views/login.php')

#     try:
#         # Find input fields and enter credentials
#         driver.find_element(By.ID, 'email').send_keys(email)
#         driver.find_element(By.ID, 'password').send_keys(password)

#         # Submit form
#         driver.find_element(By.TAG_NAME, 'form').submit()

#         # Wait for response (modify as needed)
#         time.sleep(3)

#     except Exception as e:
#         print(f"Error logging in with {email}: {e}")

# # Main Execution
# if __name__ == "__main__":
#     credentials_list = load_credentials("c:/xampp/htdocs/Projects/CAPSTONE/Bot/MOCK_DATA_test.csv")
    
#     for credential in credentials_list:
#         print(f"Trying login: {credential['email']} | {credential['password']}")

#         driver = init_driver()  # Create new driver instance
#         login(driver, credential['email'], credential['password'])

#         driver.quit()  # Close browser instance after each attempt
#         time.sleep(random.uniform(1, 5))  # Random delay to simulate human behavior

# # Import necessary libraries
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.action_chains import ActionChains
# import time
# import csv
# import random

# # Load credentials from CSV file
# def load_credentials(file_path):
#     credentials = []
#     with open(file_path, mode='r', encoding='utf-8') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             credentials.append({"email": row["email"], "password": row["password"]})
#     random.shuffle(credentials)  # Shuffle for randomness
#     return credentials

# # Initialize WebDriver (Edge in this case)
# def init_driver():
#     options = webdriver.EdgeOptions()
#     options.add_argument("--headless")  # Run in headless mode
#     options.add_argument("--disable-blink-features=AutomationControlled")  # Avoid detection
#     driver = webdriver.Edge(options=options)
#     return driver

# # Move mouse to element and click
# def move_and_click(driver, element):
#     action = ActionChains(driver)
#     action.move_to_element(element).pause(random.uniform(0.3, 1)).click().perform()

# # Simulate human-like typing
# def type_like_human(element, text):
#     for char in text:
#         element.send_keys(char)
#         time.sleep(random.uniform(0.05, 0.5))  # Varying delay between keystrokes

# # Perform login with given credentials
# def login(driver, email, password):
#     driver.get('http://localhost/Projects/CAPSTONE/Website/views/login.php')

#     try:
#         # Wait for the email field to appear
#         email_field = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'email')))
#         password_field = driver.find_element(By.ID, 'password')
#         login_button = driver.find_element(By.XPATH, '//button[@type="submit"]')  # Adjust if necessary

#         # Type credentials like a human
#         type_like_human(email_field, email)
#         type_like_human(password_field, password)

#         # Click login button instead of using form.submit()
#         move_and_click(driver, login_button)

#         # Wait for response (increase if necessary)
#         time.sleep(random.uniform(3, 6))

#     except Exception as e:
#         print(f"Error logging in with {email}: {e}")

# # Main Execution
# if __name__ == "__main__":
#     credentials_list = load_credentials("c:/xampp/htdocs/Projects/CAPSTONE/Bot/MOCK_DATA_test.csv")
    
#     for credential in credentials_list:
#         print(f"Trying login: {credential['email']} | {credential['password']}")

#         driver = init_driver()  # Create new driver instance
#         login(driver, credential['email'], credential['password'])

#         driver.quit()  # Close browser instance after each attempt
#         time.sleep(random.uniform(1, 5))  # Random delay to simulate human behavior

# Import necessary libraries
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time
import csv
import random
from fake_useragent import UserAgent

# Load credentials from CSV file
def load_credentials(file_path):
    credentials = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            credentials.append({"email": row["email"], "password": row["password"]})
    random.shuffle(credentials)  # Shuffle for randomness
    return credentials

# List of proxy IP addresses (Replace with working proxies)
# PROXY_LIST = [
#     "154.16.192.70:3128",
#     "45.79.208.5:3128",
#     "178.128.221.243:8080",
#     "64.225.8.110:9990"
# ]

# Function to initialize WebDriver with dynamic user-agent and proxy
def init_driver():
    ua = UserAgent()  # Initialize user agent generator
    user_agent = ua.random  # Get a random user agent

    # proxy = random.choice(PROXY_LIST)  # Pick a random proxy

    options = webdriver.EdgeOptions()
    options.add_argument(f"user-agent={user_agent}")  # Set random User-Agent
    # options.add_argument(f"--proxy-server={proxy}")  # Use random proxy
    options.add_argument("--disable-blink-features=AutomationControlled")  # Avoid detection
    
    print(f"Using User-Agent: {user_agent}")
    # print(f"Using Proxy: {proxy}")

    driver = webdriver.Edge(options=options)
    return driver

# Move mouse to element and click
def move_and_click(driver, element):
    action = ActionChains(driver)
    action.move_to_element(element).pause(random.uniform(0.3, 1)).click().perform()

# Simulate human-like typing
def type_like_human(element, text):
    for char in text:
        element.send_keys(char)
        time.sleep(random.uniform(0.05, 0.2))  # Varying delay between keystrokes

# Perform login with given credentials
def login(driver, email, password):
    driver.get('http://localhost/Projects/CAPSTONE/Website/views/login.php')

    try:
        # Wait for the email field to appear
        email_field = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'email')))
        password_field = driver.find_element(By.ID, 'password')
        login_button = driver.find_element(By.XPATH, '//button[@type="submit"]')  # Adjust if necessary

        # Type credentials like a human
        type_like_human(email_field, email)
        type_like_human(password_field, password)

        # Click login button instead of using form.submit()
        move_and_click(driver, login_button)

        # Wait for response (increase if necessary)
        time.sleep(random.uniform(3, 6))

    except Exception as e:
        print(f"Error logging in with {email}: {e}")

# Main Execution
if __name__ == "__main__":
    credentials_list = load_credentials("c:/xampp/htdocs/Projects/CAPSTONE/Bot/MOCK_DATA_test.csv")
    
    for credential in credentials_list:
        print(f"Trying login: {credential['email']} | {credential['password']}")

        driver = init_driver()  # Create new driver instance with a random user-agent and proxy
        login(driver, credential['email'], credential['password'])

        driver.quit()  # Close browser instance after each attempt
        time.sleep(random.uniform(1, 5))  # Random delay to simulate human behavior
