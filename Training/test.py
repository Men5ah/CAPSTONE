
# Introduction
## Imports and Setup
# Imports and Setup
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Ensure the backend is set for plt if needed
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Ensure inline plots for Kaggle
%matplotlib inline

# For reproducibility
np.random.seed(42)

print('Imports and setup complete.')
### Data Loading
# Load the dataset
df = pd.read_csv('cybersecurity_intrusion_data.csv')

# Display the first few rows
df.shape
## Data Exploration
df.head()
print('Data Types:\n', df.dtypes)
print('\nMissing Values:\n', df.isnull().sum())
## Data Cleaning and Preprocessing
# Drop any duplicates if necessary
df.drop_duplicates(inplace=True)

# For our predictive modeling, we do not need session_id (string identifier).
if 'session_id' in df.columns:
    df.drop('session_id', axis=1, inplace=True)

# Identify categorical columns
categorical_cols = ['protocol_type', 'encryption_used', 'browser_type']

# Use pd.get_dummies to encode categorical variables
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Display the transformed dataframe info
df.info()

# Check for any remaining missing values
print('Missing values after preprocessing:', df.isnull().sum().sum())
## Data Visualization
# Visual 1: Correlation Heatmap (only if 4 or more numeric columns are available)
numeric_df = df.select_dtypes(include=[np.number])
if numeric_df.shape[1] >= 4:
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()
else:
    print('Not enough numeric data for a correlation heatmap.')
# Visual 2: Pair Plot for a quick multi-dimensional view
sns.pairplot(numeric_df, diag_kind='kde')
plt.suptitle('Pair Plot of Numeric Features', y=1.02)
plt.show()
# Visual 3: Histograms of network packet size and session duration
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['network_packet_size'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Network Packet Size')

plt.subplot(1, 2, 2)
sns.histplot(df['session_duration'], kde=True, bins=30, color='salmon')
plt.title('Distribution of Session Duration')

plt.tight_layout()
plt.show()
# Visual 4: Box Plot of ip_reputation_score vs. failed_logins
plt.figure(figsize=(8, 6))
sns.boxplot(x='failed_logins', y='ip_reputation_score', data=df)
plt.title('IP Reputation Score by Number of Failed Logins')
plt.tight_layout()
plt.show()