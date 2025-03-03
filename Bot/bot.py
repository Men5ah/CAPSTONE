# import os
# import time
# import csv
# import random
# import multiprocessing
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.common.action_chains import ActionChains
# from fake_useragent import UserAgent


# EDGE_DRIVER_PATH = r"edgedriver_win64/msedgedriver.exe"

# # Load credentials from CSV
# def load_credentials(file_path):
#     credentials = []
#     with open(file_path, mode='r', encoding='utf-8') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             credentials.append({"email": row["email"], "password": row["password"]})
#     random.shuffle(credentials)  # Shuffle credentials for randomness
#     return credentials

# # NordVPN Server Locations
# NORDVPN_SERVERS = ["United States", "Canada", "Germany", "France", "United Kingdom", "Ghana", "Japan", "Nigeria", "Denmark"]

# # Function to switch VPN server
# def switch_vpn():
#     """Switches NordVPN server at intervals to avoid multiple VPN connections."""
#     server = random.choice(NORDVPN_SERVERS)
#     nordvpn_path = r'"C:/Program Files/NordVPN/"'
#     os.system(f"nordvpn -c -g {server}")
#     print(f"🔄 Switched VPN to {server}")
#     time.sleep(15)  # Allow time for VPN to fully connect

# # Function to disconnect VPN
# def disconnect_vpn():
#     """Disconnects NordVPN after the script is done."""
#     nordvpn_path = r'"C:/Program Files/NordVPN/"'
#     os.system(f"{nordvpn_path} disconnect")
#     print("🔴 VPN Disconnected.")

# # Initialize WebDriver
# def init_driver():
#     ua = UserAgent()
#     user_agent = ua.random  # Generate a random user agent

#     options = webdriver.EdgeOptions()
#     options.add_argument(f"user-agent={user_agent}")
#     options.add_argument("--disable-blink-features=AutomationControlled")
#     options.add_argument("--headless")  # Run in headless mode
#     options.add_argument("--disable-images")

#     print(f"🕵️ Using User-Agent: {user_agent}")

#     service = webdriver.EdgeService(EDGE_DRIVER_PATH)
#     driver = webdriver.Edge(options=options)
#     driver.set_page_load_timeout(600)
#     return driver

# # Move mouse to element and click
# def move_and_click(driver, element):
#     action = ActionChains(driver)
#     action.move_to_element(element).pause(random.uniform(0.3, 1)).click().perform()

# # Simulate human-like typing
# def type_like_human(element, text):
#     for char in text:
#         element.send_keys(char)
#         time.sleep(random.uniform(0.05, 0.2))  # Varying delay between keystrokes

# # Perform login
# def login(credential):
#     driver = init_driver()
#     driver.get('http://localhost/Projects/CAPSTONE/Website/views/login.php')

#     try:
#         email_field = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'email')))
#         password_field = driver.find_element(By.ID, 'password')
#         login_button = driver.find_element(By.XPATH, '//button[@type="submit"]')

#         type_like_human(email_field, credential["email"])
#         type_like_human(password_field, credential["password"])

#         move_and_click(driver, login_button)

#         time.sleep(random.uniform(3, 6))

#     except Exception as e:
#         print(f"⚠️ Error logging in with {credential['email']}: {e}")

#     driver.quit()

# # Run logins in parallel with multiprocessing
# def process_batch(credentials_batch):
#     """Runs login attempts in parallel using multiprocessing."""
#     MAX_PROCESSES = 4  # Adjust based on your system performance
#     with multiprocessing.Pool(processes=MAX_PROCESSES) as pool:
#         pool.map(login, credentials_batch)

import os
import time
import csv
import random
import multiprocessing
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from fake_useragent import UserAgent

# Constants
EDGE_DRIVER_PATH = r"edgedriver_win64/msedgedriver.exe"
NORDVPN_SERVERS = ["United States", "Canada", "Germany", "France", "United Kingdom", "Ghana", "Japan", "Nigeria", "Denmark"]
MAX_PROCESSES = 5  # Adjust based on your system performance
BATCH_SIZE = 20  # Number of credentials per batch

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load credentials from CSV
def load_credentials(file_path):
    """Load credentials from a CSV file."""
    credentials = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            credentials.append({"email": row["email"], "password": row["password"]})
    # random.shuffle(credentials)  # Shuffle credentials for randomness
    return credentials

# Switch VPN server
def switch_vpn():
    """Switch NordVPN server to a random location."""
    server = random.choice(NORDVPN_SERVERS)
    os.system(f"nordvpn -c -g {server}")
    logger.info(f"🔄 Switched VPN to {server}")
    time.sleep(15)  # Allow time for VPN to fully connect

# Disconnect VPN
def disconnect_vpn():
    """Disconnect NordVPN."""
    os.system("nordvpn disconnect")
    logger.info("🔴 VPN Disconnected.")

# Initialize WebDriver
def init_driver():
    """Initialize and return a headless Edge WebDriver."""
    ua = UserAgent()
    user_agent = ua.random  # Generate a random user agent

    options = webdriver.EdgeOptions()
    options.add_argument(f"user-agent={user_agent}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--disable-images")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    logger.info(f"🕵️ Using User-Agent: {user_agent}")

    service = webdriver.EdgeService(EDGE_DRIVER_PATH)
    driver = webdriver.Edge(service=service, options=options)
    driver.set_page_load_timeout(1200)
    return driver

# Move mouse to element and click
def move_and_click(driver, element):
    """Simulate human-like mouse movement and click."""
    action = ActionChains(driver)
    action.move_to_element(element).pause(random.uniform(0.3, 1)).click().perform()

# Simulate human-like typing
def type_like_human(element, text):
    """Simulate human-like typing with random delays."""
    for char in text:
        element.send_keys(char)
        time.sleep(random.uniform(0.05, 0.2))  # Varying delay between keystrokes

# Perform login
def login(credential):
    """Perform login using the provided credentials."""
    driver = init_driver()
    try:
        driver.get('http://localhost/Projects/CAPSTONE/Website/views/login.php')

        email_field = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'email')))
        password_field = driver.find_element(By.ID, 'password')
        login_button = driver.find_element(By.XPATH, '//button[@type="submit"]')

        type_like_human(email_field, credential["email"])
        type_like_human(password_field, credential["password"])

        move_and_click(driver, login_button)

        time.sleep(random.uniform(3, 6))  # Simulate human-like delay

    except Exception as e:
        logger.error(f"⚠️ Error logging in with {credential['email']}: {e}")
    finally:
        driver.quit()

# Process credentials in batches
def process_batch(credentials_batch):
    """Process a batch of credentials in parallel."""
    with multiprocessing.Pool(processes=MAX_PROCESSES) as pool:
        pool.map(login, credentials_batch)

# Main function
def main():
    """Main function to execute the script."""
    credentials = load_credentials("C:/xampp/htdocs/Projects/CAPSTONE/Bot/credentials/cred1.csv")
    total_batches = (len(credentials) // BATCH_SIZE) + 1

    for i in range(total_batches):
        batch_start = i * BATCH_SIZE
        batch_end = batch_start + BATCH_SIZE
        batch = credentials[batch_start:batch_end]

        logger.info(f"🚀 Processing batch {i + 1}/{total_batches} with {len(batch)} credentials")
        switch_vpn()  # Switch VPN before processing each batch
        process_batch(batch)
        time.sleep(5)  # Add a delay between batches

    disconnect_vpn()  # Disconnect VPN after all batches are processed

if __name__ == "__main__":
    main()