import numpy as np
import pandas as pd
import random

# Set random seed for reproducibility
np.random.seed(42)

# Number of records
num_humans = 1000
num_bots = 500

# Helper functions to generate features
def generate_mouse_speed(is_bot):
    return np.random.normal(300 if is_bot else 200, 50 if is_bot else 75)  # bots have faster, consistent speeds

def generate_typing_speed(is_bot):
    return np.random.normal(10 if is_bot else 5, 2 if is_bot else 1)  # bots type faster and more consistently

def generate_click_behavior(is_bot):
    return random.choice(["Random", "Predictable"]) if is_bot else "Human-like"

def generate_page_dwell_time(is_bot):
    return np.random.uniform(0.5, 2.0) if is_bot else np.random.uniform(10, 50)

def generate_login_attempts(is_bot):
    return np.random.randint(2, 6) if is_bot else np.random.randint(0, 2)

def generate_browser_type():
    return random.choice(["Chrome", "Firefox", "Safari", "Edge", "Opera"])

def generate_ip(is_bot):
    return f"192.168.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}" if not is_bot else f"10.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}"

def generate_geolocation(is_bot):
    return random.choice(["USA", "Europe", "Asia"]) if is_bot else random.choice(["USA", "Canada", "Australia"])

def generate_browser_reputation_score(is_bot):
    return np.random.uniform(0.1, 0.5) if is_bot else np.random.uniform(0.7, 1.0)

def generate_latency(is_bot):
    return np.random.uniform(50, 150) if is_bot else np.random.uniform(10, 50)

# Generate the dataset
def generate_dataset(num_humans, num_bots):
    data = []
    for i in range(num_humans + num_bots):
        is_bot = 1 if i >= num_humans else 0
        record = {
            "mouse_speed": generate_mouse_speed(is_bot),
            "typing_speed": generate_typing_speed(is_bot),
            "click_behavior": generate_click_behavior(is_bot),
            "page_dwell_time": generate_page_dwell_time(is_bot),
            "login_attempts": generate_login_attempts(is_bot),
            "browser_type": generate_browser_type(),
            "ip_address": generate_ip(is_bot),
            "login_location": generate_geolocation(is_bot),
            "browser_reputation_score": generate_browser_reputation_score(is_bot),
            "latency": generate_latency(is_bot),
            "is_bot": is_bot
        }
        data.append(record)
    return pd.DataFrame(data)

# Generate and save the dataset
dataset = generate_dataset(num_humans, num_bots)
dataset.to_csv("synthetic_user_bot_data.csv", index=False)

print(f"Dataset generated with {len(dataset)} records.")