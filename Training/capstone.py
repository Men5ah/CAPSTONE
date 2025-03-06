# Capstone Model Training
## Data Preprocessing
## Data Visualization
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Set up visualization style
sns.set(style="whitegrid")
# Load the dataset
df = pd.read_csv('cybersecurity_intrusion_data.csv')

# Display the first few rows
df.head()
## Exploratory Data Analysis
df = df.drop('session_id', axis=1)
# Check for missing values
df.isnull().sum()
# Summary statistics for numerical features
df.describe()
# Class distribution
df['attack_detected'].value_counts()

# Visualize class distribution
sns.countplot(x='attack_detected', data=df)
plt.title('Class Distribution')
plt.show()
## Data Preprocessing
# Check for missing values again
df.isnull().sum()

# If there are missing values, impute them
# For numerical features, use mean or median
# For categorical features, use mode
numerical_features = ['network_packet_size', 'session_duration', 'ip_reputation_score']
categorical_features = ['protocol_type', 'encryption_used', 'browser_type']

# Impute missing values
df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())
df[categorical_features] = df[categorical_features].fillna(df[categorical_features].mode().iloc[0])
# Encode categorical features
df_encoded = pd.get_dummies(df, columns=['protocol_type', 'encryption_used', 'browser_type'], drop_first=False)

# Compute the correlation matrix
corr_matrix = df_encoded.corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix (After Encoding Categorical Features)')
plt.show()

df_encoded
# Split into features and target
X = df.drop('attack_detected', axis=1)
y = df['attack_detected']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check shapes
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)
## Feature Selection and Engineering
### Pairplots
# Add the target variable to the DataFrame
df_pairplot = X_train.copy()
df_pairplot['attack_detected'] = y_train

# Plot pairplot
sns.pairplot(df_pairplot, hue='attack_detected', diag_kind='kde')
plt.show()
### Feature Importance and Mutual Information
from sklearn.ensemble import RandomForestClassifier

# Split into features and target
X = df_encoded.drop('attack_detected', axis=1)
y = df_encoded['attack_detected']

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance from Random Forest')
plt.show()
from sklearn.feature_selection import mutual_info_classif

# Compute Mutual Information
mi_scores = mutual_info_classif(df_encoded.drop('attack_detected', axis=1), df_encoded['attack_detected'])

# Create a DataFrame for visualization
mi_df = pd.DataFrame({
    'Feature': df_encoded.drop('attack_detected', axis=1).columns,
    'MI Score': mi_scores
})

# Sort by MI score
mi_df = mi_df.sort_values(by='MI Score', ascending=False)

# Plot MI scores
plt.figure(figsize=(10, 6))
sns.barplot(x='MI Score', y='Feature', data=mi_df)
plt.title('Mutual Information Scores')
plt.show()
### Chi-Square
from sklearn.feature_selection import chi2

# Identify one-hot encoded categorical features
categorical_encoded_features = [col for col in df_encoded.columns if col.startswith(('protocol_type', 'encryption_used', 'browser_type'))]

# Perform Chi-Square test
chi_scores, p_values = chi2(df_encoded[categorical_encoded_features], df_encoded['attack_detected'])

# Create a DataFrame for visualization
chi_df = pd.DataFrame({
    'Feature': categorical_encoded_features,
    'Chi-Square Score': chi_scores,
    'P-Value': p_values
})

# Sort by Chi-Square score
chi_df = chi_df.sort_values(by='Chi-Square Score', ascending=False)

# Plot Chi-Square scores
plt.figure(figsize=(10, 6))
sns.barplot(x='Chi-Square Score', y='Feature', data=chi_df)
plt.title('Chi-Square Scores for Categorical Features')
plt.show()
### Point-Biserial and ANOVA F-Test
from scipy.stats import pointbiserialr

# Select numerical features
numerical_features = ['network_packet_size', 'session_duration', 'ip_reputation_score', 'failed_logins', 'unusual_time_access']

# Compute Point-Biserial Correlation
pb_scores = [pointbiserialr(X_train[feature], y_train)[0] for feature in numerical_features]

# Create a DataFrame for visualization
pb_df = pd.DataFrame({
    'Feature': numerical_features,
    'Point-Biserial Correlation': pb_scores
})

# Sort by correlation
pb_df = pb_df.sort_values(by='Point-Biserial Correlation', ascending=False)

# Plot Point-Biserial Correlation
plt.figure(figsize=(10, 6))
sns.barplot(x='Point-Biserial Correlation', y='Feature', data=pb_df)
plt.title('Point-Biserial Correlation for Numerical Features')
plt.show()
from sklearn.feature_selection import f_classif

# Perform ANOVA F-Test
f_scores, p_values = f_classif(X_train[numerical_features], y_train)

# Create a DataFrame for visualization
anova_df = pd.DataFrame({
    'Feature': numerical_features,
    'F-Score': f_scores,
    'P-Value': p_values
})

# Sort by F-Score
anova_df = anova_df.sort_values(by='F-Score', ascending=False)

# Plot F-Scores
plt.figure(figsize=(10, 6))
sns.barplot(x='F-Score', y='Feature', data=anova_df)
plt.title('ANOVA F-Scores for Numerical Features')
plt.show()