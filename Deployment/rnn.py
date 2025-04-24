from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import logging
from typing import Dict, Any, Union, List
from tensorflow.keras.models import load_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the RNN model, preprocessor, and feature info
def load_artifacts() -> Union[Dict[str, Any], None]:
    """Load model, preprocessor, and feature info artifacts."""
    try:
        artifacts = {}

        # Load keras model
        model_path = os.environ.get('RNN_MODEL_PATH', 'rnn_model_noisy.keras')
        artifacts['model'] = load_model(model_path)
        logger.info(f"RNN model loaded successfully from {model_path}")

        # Load preprocessor
        preprocessor_path = os.environ.get('PREPROCESSOR_PATH', 'preprocessor_noisy.pkl')
        artifacts['preprocessor'] = joblib.load(preprocessor_path)
        logger.info(f"Preprocessor loaded successfully from {preprocessor_path}")

        # Load feature info
        feature_info_path = os.environ.get('FEATURE_INFO_PATH', 'feature_info_noisy.pkl')
        artifacts['feature_info'] = joblib.load(feature_info_path)
        logger.info(f"Feature info loaded successfully from {feature_info_path}")

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
            'message': 'Model, preprocessor, or feature info not loaded',
            'healthy': False,
            'timestamp': datetime.now().isoformat()
        }), 500

    return jsonify({
        'status': 'ok',
        'healthy': True,
        'model_loaded': True,
        'preprocessor_loaded': True,
        'feature_info_loaded': True,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict() -> Dict[str, Any]:
    """Endpoint to make predictions using the RNN model with proper preprocessing."""
    if artifacts is None:
        return jsonify({
            'status': 'error',
            'message': 'Model, preprocessor, or feature info not loaded',
            'timestamp': datetime.now().isoformat()
        }), 500

    try:
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

        # Validate required features from feature_info
        required_features = artifacts['feature_info'].get('features', None)
        if required_features is None:
            return jsonify({
                'status': 'error',
                'message': 'Feature info does not contain required features list',
                'timestamp': datetime.now().isoformat()
            }), 500

        missing_features = set(required_features) - set(input_data.columns)
        if missing_features:
            return jsonify({
                'status': 'error',
                'message': f'Missing required features: {missing_features}',
                'required_features': required_features,
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

        # The RNN model likely expects 3D input: (samples, timesteps, features)
        # Assuming timesteps=1 for tabular data, reshape accordingly
        try:
            processed_data_reshaped = np.expand_dims(processed_data, axis=1)
        except Exception as e:
            logger.error(f"Error reshaping data for RNN input: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Error reshaping data for RNN input: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }), 500

        # Make predictions
        predictions_prob = artifacts['model'].predict(processed_data_reshaped)
        # If output is probability, convert to class labels (assuming binary classification)
        predictions = (predictions_prob > 0.5).astype(int).flatten()

        # Prepare response
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, predictions_prob.flatten())):
            results.append({
                'prediction': int(pred),
                'probability': float(prob),
                'confidence': f"{prob*100:.2f}%"
            })

        response = {
            'status': 'success',
            'predictions': results,
            'model_type': 'RNN (trained on noisy data)',
            'preprocessing': 'Custom preprocessor loaded from preprocessor_noisy.pkl',
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

        # Get model summary as string
        import io
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_str = stream.getvalue()

        return jsonify({
            'status': 'success',
            'model_type': 'Keras RNN model',
            'model_summary': summary_str,
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
    if artifacts is None or 'feature_info' not in artifacts:
        return jsonify({
            'status': 'error',
            'message': 'Feature info not loaded',
            'timestamp': datetime.now().isoformat()
        }), 500

    try:
        feature_info = artifacts['feature_info']
        return jsonify({
            'status': 'success',
            'feature_info': feature_info,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting feature info: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
