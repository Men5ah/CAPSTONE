import os
import time
import csv
import random
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from fake_useragent import UserAgent

# Constants
EDGE_DRIVER_PATH = r"edgedriver_win64/msedgedriver.exe"
NORDVPN_SERVERS = ["United States", "Canada", "Germany", "France", "United Kingdom", "Ghana", "Japan", "Nigeria", "Denmark"]

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

# Initialize WebDriver with console tab open
def init_driver():
    """Initialize and return an Edge WebDriver in fullscreen with Console tab open."""
    ua = UserAgent()
    user_agent = ua.random

    options = webdriver.EdgeOptions()
    options.add_argument(f"user-agent={user_agent}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-images")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--auto-open-devtools-for-tabs")  # Ensure devtools open
    
    service = webdriver.EdgeService(EDGE_DRIVER_PATH)
    driver = webdriver.Edge(service=service, options=options)
    
    # Maximize to fullscreen
    driver.maximize_window()
    
    # Use keyboard shortcut to focus console (Ctrl+Shift+J)
    try:
        actions = ActionChains(driver)
        actions.key_down(Keys.CONTROL).key_down(Keys.SHIFT).send_keys('j').key_up(Keys.SHIFT).key_up(Keys.CONTROL).perform()
        time.sleep(1)  # Allow time for the console to focus
    except Exception as e:
        logger.warning(f"⚠️ Could not focus console tab: {e}")
    
    driver.set_page_load_timeout(150)
    return driver

# Move mouse to element and click with added randomness in pause duration
def move_and_click(driver, element):
    """Simulate human-like mouse movement and click with random pause duration."""
    action = ActionChains(driver)
    random_pause = random.uniform(0.5, 2.0)
    action.move_to_element(element).pause(random_pause).click().perform()

# Simulate human-like typing
def type_like_human(element, text):
    """Simulate human-like typing with random delays."""
    for char in text:
        element.send_keys(char)
        time.sleep(random.uniform(0.15, 0.2))  # Varying delay between keystrokes

# Perform login
def login(credential):
    """Perform login using the provided credentials."""
    driver = init_driver()
    try:
        driver.get('http://localhost/Projects/CAPSTONE/Website/views/login.php')

        # Wait for and fill email field
        email_field = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'email')))
        type_like_human(email_field, credential["email"])
        
        # Fill password field
        password_field = driver.find_element(By.ID, 'password')
        type_like_human(password_field, credential["password"])
        
        # Find and click login button
        login_button = driver.find_element(By.ID, 'login_button')  # Adjust this selector as needed
        move_and_click(driver, login_button)

        time.sleep(random.uniform(3, 6))  # Simulate human-like delay

    except Exception as e:
        logger.error(f"⚠️ Error logging in with {credential['email']}: {e}")
        # Take screenshot on error
        driver.save_screenshot(f"error_{credential['email']}.png")
    finally:
        driver.quit()

# Main function
def main():
    """Main function to execute the script."""
    credentials = load_credentials("C:/xampp/htdocs/Projects/CAPSTONE/Bot/credentials/cred1.csv")

    for credential in credentials:
        logger.info(f"🚀 Processing login for {credential['email']}")
        switch_vpn()  # Optionally switch VPN before each login
        login(credential)
        time.sleep(5)  # Add a delay between logins

    disconnect_vpn()  # Disconnect VPN after all logins are processed

if __name__ == "__main__":
    main()