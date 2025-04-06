import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define number of users and total login records
num_users = 2000
num_records = 50
bot_percentage = 0.45  # 45% of users are bots

# Generate user base
users = [f'user_{i}' for i in range(1, num_users + 1)]
num_bots = int(num_users * bot_percentage)
bot_users = set(random.sample(users, num_bots))

# Define feature distributions
def generate_login_attempts(is_bot):
    return np.random.randint(1, 6) if not is_bot else np.random.randint(3, 10)

def generate_failed_logins(is_bot):
    return np.random.randint(0, 3) if not is_bot else np.random.randint(1, 6)

def generate_unusual_time_access():
    return np.random.choice(["False", "True"], p=[0.8, 0.2])

def generate_ip_rep_score(is_bot):
    return round(np.random.uniform(0, 0.5), 2) if not is_bot else round(np.random.uniform(0.5, 1), 2)

def generate_browser_type():
    return np.random.choice(['Chrome', 'Firefox', 'Edge', 'Safari', 'Other'], p=[0.5, 0.2, 0.15, 0.1, 0.05])

def generate_new_device_login():
    return np.random.choice(["False", "True"], p=[0.9, 0.1])

def generate_session_duration_deviation(is_bot):
    return round(np.random.uniform(0, 2), 2) if not is_bot else round(np.random.uniform(2, 5), 2)

def generate_network_packet_size_variance(is_bot):
    return round(np.random.uniform(0, 1), 2) if not is_bot else round(np.random.uniform(1, 3), 2)

def generate_mouse_speed(is_bot):
    return round(np.random.uniform(0.5, 2), 2) if not is_bot else round(np.random.uniform(2, 5), 2)

def generate_typing_speed(is_bot):
    return round(np.random.uniform(40, 80), 2) if not is_bot else round(np.random.uniform(10, 40), 2)

def generate_day_of_week():
    return np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

def generate_time_of_day():
    return np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'])

# Generate login records
data = []
start_date = datetime.now() - timedelta(days=30)

for _ in range(num_records):
    user = random.choice(users)
    is_bot = user in bot_users
    login_date = start_date + timedelta(days=np.random.randint(0, 30))
    
    data.append([
        user,
        generate_login_attempts(is_bot),
        generate_failed_logins(is_bot),
        generate_unusual_time_access(),
        generate_ip_rep_score(is_bot),
        generate_browser_type(),
        generate_new_device_login(),
        generate_session_duration_deviation(is_bot),
        generate_network_packet_size_variance(is_bot),
        generate_mouse_speed(is_bot),
        generate_typing_speed(is_bot),
        generate_day_of_week(),
        generate_time_of_day(),
        is_bot  # Target variable (1 = bot, 0 = human)
    ])

# Create DataFrame
columns = [
    'user_id', 'login_attempts', 'failed_logins', 'unusual_time_access', 'ip_rep_score', 'browser_type', 
    'new_device_login', 'session_duration_deviation', 'network_packet_size_variance', 'mouse_speed', 
    'typing_speed', 'day_of_week', 'time_of_day', 'is_bot'
]
df = pd.DataFrame(data, columns=columns)

# Save dataset
df.to_csv('Dataset/clean_test_test.csv', index=False)

print("Dataset generation complete. Saved as 'clean_test.csv'.")





import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define number of users and total login records
num_users = 2000
num_records = 50
bot_percentage = 0.45  # 45% of users are bots

# Generate user base
users = [f'user_{i}' for i in range(1, num_users + 1)]
num_bots = int(num_users * bot_percentage)
bot_users = set(random.sample(users, num_bots))

# Define feature distributions with noise
def generate_login_attempts(is_bot):
    base = np.random.randint(1, 6) if not is_bot else np.random.randint(3, 10)
    return max(0, base + np.random.randint(-2, 3))  # Add small noise

def generate_failed_logins(is_bot):
    base = np.random.randint(0, 3) if not is_bot else np.random.randint(1, 6)
    return max(0, base + np.random.randint(-1, 2))  # Noise

def generate_unusual_time_access():
    return np.random.choice(["False", "True"], p=[0.75, 0.25])  # Slightly more noise

def generate_ip_rep_score(is_bot):
    base = np.random.uniform(0, 0.5) if not is_bot else np.random.uniform(0.5, 1)
    return round(base + np.random.uniform(-0.1, 0.1), 2)  # Add slight shift

def generate_browser_type():
    return np.random.choice(['Chrome', 'Firefox', 'Edge', 'Safari', 'Other'], p=[0.45, 0.25, 0.15, 0.1, 0.05])

def generate_new_device_login():
    return np.random.choice(["False", "True"], p=[0.85, 0.15])  # More noise

def generate_session_duration_deviation(is_bot):
    base = np.random.uniform(0, 2) if not is_bot else np.random.uniform(2, 5)
    return round(base + np.random.uniform(-0.5, 0.5), 2)  # Perturbation

def generate_network_packet_size_variance(is_bot):
    base = np.random.uniform(0, 1) if not is_bot else np.random.uniform(1, 3)
    return round(base + np.random.uniform(-0.3, 0.3), 2)  # Noise

def generate_mouse_speed(is_bot):
    base = np.random.uniform(0.5, 2) if not is_bot else np.random.uniform(2, 5)
    return round(base + np.random.uniform(-0.5, 0.5), 2)

def generate_typing_speed(is_bot):
    base = np.random.uniform(40, 80) if not is_bot else np.random.uniform(10, 40)
    return round(base + np.random.uniform(-5, 5), 2)

def generate_day_of_week():
    return np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

def generate_time_of_day():
    return np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'])

# Generate login records
data = []
start_date = datetime.now() - timedelta(days=30)

for _ in range(num_records):
    user = random.choice(users)
    is_bot = user in bot_users
    login_date = start_date + timedelta(days=np.random.randint(0, 30))
    
    record = [
        user,
        generate_login_attempts(is_bot),
        generate_failed_logins(is_bot),
        generate_unusual_time_access(),
        generate_ip_rep_score(is_bot),
        generate_browser_type(),
        generate_new_device_login(),
        generate_session_duration_deviation(is_bot),
        generate_network_packet_size_variance(is_bot),
        generate_mouse_speed(is_bot),
        generate_typing_speed(is_bot),
        generate_day_of_week(),
        generate_time_of_day(),
        is_bot  # Target variable (1 = bot, 0 = human)
    ]
    
    # Introduce missing values randomly
    if random.random() < 0.10:  # 10% missing data
        missing_index = np.random.randint(1, len(record) - 1)
        record[missing_index] = None

    # Flip labels for some cases to simulate misclassification
    if random.random() < 0.8:  # 8% label noise
        record[-1] = not record[-1]
    
    data.append(record)

# Create DataFrame
columns = [
    'user_id', 'login_attempts', 'failed_logins', 'unusual_time_access', 'ip_rep_score', 'browser_type', 
    'new_device_login', 'session_duration_deviation', 'network_packet_size_variance', 'mouse_speed', 
    'typing_speed', 'day_of_week', 'time_of_day', 'is_bot'
]
df = pd.DataFrame(data, columns=columns)

# Save dataset
df.to_csv('Dataset/noisy_test_test.csv', index=False)

print("Dataset generation complete with noise. Saved as 'noisy_test.csv'.")
