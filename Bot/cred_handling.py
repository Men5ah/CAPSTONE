import csv
import random

# Load credentials from CSV file
def load_credentials(file_path):
    credentials = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            credentials.append({"email": row["email"], "password": row["password"]})
    return credentials

# Rotate credentials (shuffle for randomness)
def get_next_credential(credentials, index):
    return credentials[index % len(credentials)]  # Ensures it loops through the list

# Example usage
credentials_list = load_credentials("c:/xampp/htdocs/Projects/CAPSTONE/Bot/credentials.csv")
index = 0  # Track which credential is being used

while index < len(credentials_list):
    credential = get_next_credential(credentials_list, index)
    print(f"Trying login: {credential['email']} | {credential['password']}")
    index += 1
