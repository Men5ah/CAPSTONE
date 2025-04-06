from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import logging
from typing import Dict, Any, Union, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(_name_)

app = Flask(_name_)

# Constants for expected features
NUMERICAL_FEATURES = [
    'login_attempts', 'failed_logins', 'ip_rep_score',
    'session_duration_deviation', 'network_packet_size_variance',
    'mouse_speed', 'typing_speed'
]

CATEGORICAL_FEATURES = [
    'unusual_time_access', 'browser_type',
    'new_device_login', 'day_of_week', 'time_of_day'
]

# Load the GBM model and preprocessor
def load_artifacts() -> Union[Dict[str, Any], None]:
    """Load model and preprocessor artifacts."""
    try:
        artifacts = {}
        
        # Load model
        model_path = os.environ.get('MODEL_PATH', 'gbm_noisy_model.pkl')
        artifacts['model'] = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        
        # Load preprocessor
        preprocessor_path = os.environ.get('PREPROCESSOR_PATH', 'noisy_data_preprocessor.pkl')
        artifacts['preprocessor'] = joblib.load(preprocessor_path)
        logger.info(f"Preprocessor loaded successfully from {preprocessor_path}")
        
        return artifacts
    except Exception as e:
        logger.error(f"Error loading artifacts: {str(e)}")
        return None

artifacts = load_artifacts()

@app.route('/health', methods=['GET'])
def health_check() -> Dict[str, Any]:
    """Endpoint to check if the API is up and running with all dependencies."""
    if artifacts is None:
        return jsonify({
            'status': 'error',
            'message': 'Model or preprocessor not loaded',
            'healthy': False,
            'timestamp': datetime.now().isoformat()
        }), 500
    
    return jsonify({
        'status': 'ok',
        'healthy': True,
        'model_loaded': True,
        'preprocessor_loaded': True,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict() -> Dict[str, Any]:
    """Endpoint to make predictions using the GBM model with proper preprocessing."""
    if artifacts is None:
        return jsonify({
            'status': 'error',
            'message': 'Model or preprocessor not loaded',
            'timestamp': datetime.now().isoformat()
        }), 500
    
    try:
        # Get input data from request
        content = request.json
        
        if not content:
            return jsonify({
                'status': 'error',
                'message': 'No data provided',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Accept both single record and batch processing
        if isinstance(content, list):
            input_data = pd.DataFrame(content)
        elif isinstance(content, dict):
            input_data = pd.DataFrame([content])
        else:
            return jsonify({
                'status': 'error',
                'message': 'Input data must be JSON object or array',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        logger.info(f"Received prediction request with {len(input_data)} samples")
        
        # Validate required features
        missing_features = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES) - set(input_data.columns)
        if missing_features:
            return jsonify({
                'status': 'error',
                'message': f'Missing required features: {missing_features}',
                'required_features': NUMERICAL_FEATURES + CATEGORICAL_FEATURES,
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Preprocess the input data
        try:
            processed_data = artifacts['preprocessor'].transform(input_data)
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Preprocessing failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Make predictions
        predictions = artifacts['model'].predict(processed_data)
        probabilities = artifacts['model'].predict_proba(processed_data)
        
        # Prepare response
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'prediction': int(pred),
                'is_bot': bool(pred),
                'probability': float(prob[1]),  # Probability of being a bot
                'confidence': f"{prob[1]*100:.2f}%"
            })
        
        response = {
            'status': 'success',
            'predictions': results,
            'model_type': 'GBM (trained on noisy data)',
            'preprocessing': 'Median imputation for numerical, mode for categorical',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info() -> Dict[str, Any]:
    """Endpoint to get detailed information about the model."""
    if artifacts is None or 'model' not in artifacts:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded',
            'timestamp': datetime.now().isoformat()
        }), 500
    
    try:
        model = artifacts['model']
        
        # Get feature importances
        try:
            importances = model.feature_importances_
            # Get feature names from preprocessor
            num_feature_names = NUMERICAL_FEATURES
            cat_feature_names = artifacts['preprocessor'].named_transformers_['pipeline-2'].named_steps['onehotencoder'].get_feature_names_out(CATEGORICAL_FEATURES)
            all_feature_names = list(num_feature_names) + list(cat_feature_names)
            
            importance_dict = dict(zip(all_feature_names, importances))
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
        except Exception as e:
            logger.warning(f"Could not get feature importances: {str(e)}")
            importance_dict = {}
        
        # Get model parameters
        try:
            params = model.get_params()
            # Convert numpy types to native Python types for JSON serialization
            params = {k: v.item() if hasattr(v, 'item') else str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v 
                     for k, v in params.items()}
        except Exception as e:
            logger.warning(f"Could not get model parameters: {str(e)}")
            params = {}
        
        return jsonify({
            'status': 'success',
            'model_type': 'GradientBoostingClassifier',
            'training_data': 'Noisy dataset with imputation',
            'feature_importances': importance_dict,
            'parameters': params,
            'preprocessing_steps': {
                'numerical': ['median_imputation', 'standard_scaling'],
                'categorical': ['mode_imputation', 'one_hot_encoding']
            },
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/features', methods=['GET'])
def feature_info() -> Dict[str, Any]:
    """Endpoint to get information about expected features."""
    return jsonify({
        'status': 'success',
        'numerical_features': NUMERICAL_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'categorical_options': {
            'unusual_time_access': ['True', 'False'],
            'browser_type': ['Chrome', 'Firefox', 'Safari', 'Edge', 'Other'],
            'new_device_login': ['True', 'False'],
            'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            'time_of_day': ['Morning', 'Afternoon', 'Evening', 'Night']
        },
        'timestamp': datetime.now().isoformat()
    })

if _name_ == '_main_':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)