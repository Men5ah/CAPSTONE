import joblib

# Load feature info to check its structure
feature_info_path = 'C:/xampp/htdocs/Projects/CAPSTONE/Deployment/feature_info_noisy.pkl'
feature_info = joblib.load(feature_info_path)
print(feature_info)  # Check the structure of the loaded feature info