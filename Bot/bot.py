import os
import time
import csv
import random
import multiprocessing
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from fake_useragent import UserAgent

# Load credentials from CSV
def load_credentials(file_path):
    credentials = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            credentials.append({"email": row["email"], "password": row["password"]})
    random.shuffle(credentials)  # Shuffle credentials for randomness
    return credentials

# NordVPN Server Locations
NORDVPN_SERVERS = ["United States", "Canada", "Germany", "France", "United Kingdom", "Ghana", "Japan", "Nigeria", "Denmark"]

# Function to switch VPN server
def switch_vpn():
    """Switches NordVPN server at intervals to avoid multiple VPN connections."""
    server = random.choice(NORDVPN_SERVERS)
    nordvpn_path = r'"C:/Program Files/NordVPN/"'
    os.system(f"nordvpn -c -g {server}")
    print(f"🔄 Switched VPN to {server}")
    time.sleep(15)  # Allow time for VPN to fully connect

# Function to disconnect VPN
def disconnect_vpn():
    """Disconnects NordVPN after the script is done."""
    nordvpn_path = r'"C:/Program Files/NordVPN/"'
    os.system(f"{nordvpn_path} disconnect")
    print("🔴 VPN Disconnected.")

# Initialize WebDriver
def init_driver():
    ua = UserAgent()
    user_agent = ua.random  # Generate a random user agent

    options = webdriver.EdgeOptions()
    options.add_argument(f"user-agent={user_agent}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--disable-images")

    print(f"🕵️ Using User-Agent: {user_agent}")

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

# Perform login
def login(credential):
    driver = init_driver()
    driver.get('http://localhost/Projects/CAPSTONE/Website/views/login.php')

    try:
        email_field = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'email')))
        password_field = driver.find_element(By.ID, 'password')
        login_button = driver.find_element(By.XPATH, '//button[@type="submit"]')

        type_like_human(email_field, credential["email"])
        type_like_human(password_field, credential["password"])

        move_and_click(driver, login_button)

        time.sleep(random.uniform(3, 6))

    except Exception as e:
        print(f"⚠️ Error logging in with {credential['email']}: {e}")

    driver.quit()

# Run logins in parallel
def process_batch(credentials_batch):
    """Runs login attempts in parallel using multiprocessing."""
    with multiprocessing.Pool(processes=len(credentials_batch)) as pool:
        pool.map(login, credentials_batch)

# Main Execution
if __name__ == "__main__":
    credentials_list = load_credentials("c:/xampp/htdocs/Projects/CAPSTONE/Bot/credentials1-5.csv")

    BATCH_SIZE = 10  # Number of logins before switching VPN

    # switch_vpn()  # 🔄 Connect to VPN initially

    for i in range(0, len(credentials_list), BATCH_SIZE):
        batch = credentials_list[i:i + BATCH_SIZE]
        print(f"🔹 Processing batch {i // BATCH_SIZE + 1} with {len(batch)} accounts")

        process_batch(batch)  # Run logins in parallel

        # if i + BATCH_SIZE < len(credentials_list):  # Don't switch VPN on the last batch
            # print("🔄 Switching VPN after batch...")
            # switch_vpn()

    # disconnect_vpn()  # Disconnect VPN when script is done
