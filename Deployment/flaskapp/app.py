from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from datetime import datetime
import logging
import json
from typing import Dict, Any, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def load_artifacts() -> Union[Dict[str, Any], None]:
    """Load model, preprocessor, and feature info artifacts."""
    artifacts = {}
    
    # Load model - handling both SavedModel and H5 formats
    model_path = 'Deployment/rnn_model_noisy'
    model_h5_path = 'Deployment/rnn_model_noisy.h5'
    
    try:
        # First try loading as H5 model if it exists
        if os.path.exists(model_h5_path):
            try:
                from tensorflow.keras.models import load_model
                artifacts['model'] = load_model(model_h5_path, compile=True)
                logger.info(f"Keras H5 model loaded successfully from {model_h5_path}")
                artifacts['model_type'] = 'keras_h5'
            except Exception as e:
                logger.error(f"Error loading H5 model: {str(e)}")
                return None
        
        # If H5 not found, try loading as SavedModel
        elif os.path.exists(model_path):
            try:
                # Load as SavedModel
                artifacts['model'] = tf.saved_model.load(model_path)
                logger.info(f"SavedModel loaded successfully from {model_path}")
                artifacts['model_type'] = 'saved_model'
            except Exception as e:
                logger.error(f"Error loading SavedModel: {str(e)}")
                return None
        else:
            logger.error(f"No model found at {model_path} or {model_h5_path}")
            return None
    except Exception as e:
        logger.error(f"Error in model loading process: {str(e)}")
        return None
        
    # Load preprocessor
    preprocessor_path = 'Deployment/preprocessor_noisy.pkl'
    try:
        if os.path.exists(preprocessor_path):
            artifacts['preprocessor'] = joblib.load(preprocessor_path)
            logger.info(f"Preprocessor loaded successfully from {preprocessor_path}")
        else:
            logger.error(f"Preprocessor file not found at {preprocessor_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading preprocessor: {str(e)}")
        return None
        
    # Load feature info
    feature_info_path = 'Deployment/feature_info_noisy.pkl'
    try:
        if os.path.exists(feature_info_path):
            artifacts['feature_info'] = joblib.load(feature_info_path)
            logger.info(f"Feature info loaded successfully from {feature_info_path}")
        else:
            logger.error(f"Feature info file not found at {feature_info_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading feature info: {str(e)}")
        return None

    return artifacts

def predict_with_model(model, data, model_type):
    """Handle prediction for different model types."""
    if model_type == 'keras_h5':
        return model.predict(data)
    elif model_type == 'saved_model':
        # For SavedModel format
        try:
            # Try using the serving signature
            infer = model.signatures['serving_default']
            input_name = list(infer.structured_input_signature[1].keys())[0]
            result = infer(**{input_name: tf.convert_to_tensor(data, dtype=tf.float32)})
            output_name = list(result.keys())[0]
            return result[output_name].numpy()
        except Exception as e2:
            logger.warning(f"Prediction attempt with serving signature failed: {e2}")
            # Last attempt - try to find a function called predict
            try:
                if hasattr(model, 'predict'):
                    return model.predict(data)
                else:
                    raise AttributeError("Model has no predict method and other attempts failed")
            except Exception as e3:
                logger.error(f"All prediction methods failed: {e3}")
                return None

def make_prediction(input_data, artifacts):
    """Process input data and make prediction."""
    try:
        # Parse input data
        content = pd.read_json(input_data)

        # Retrieve numerical and categorical features from feature info
        numerical_features = artifacts['feature_info'].get('numerical_features', [])
        categorical_features = artifacts['feature_info'].get('categorical_features', [])

        # Combine numerical and categorical features for validation
        required_features = numerical_features + categorical_features

        # Validate required features
        missing_features = set(required_features) - set(content.columns)
        if missing_features:
            return {"error": f"Missing required features: {missing_features}"}, 400
        
        # Preprocess the input data
        processed_data = artifacts['preprocessor'].transform(content)

        # Reshape for RNN input
        processed_data_reshaped = np.expand_dims(processed_data, axis=1)

        # Make predictions using appropriate method
        predictions_prob = predict_with_model(
            artifacts['model'], 
            processed_data_reshaped, 
            artifacts['model_type']
        )
        
        if predictions_prob is not None:
            # Make sure predictions are in the right format
            if len(predictions_prob.shape) > 1 and predictions_prob.shape[1] > 1:
                # For multi-class output, get the class with highest probability
                predictions = np.argmax(predictions_prob, axis=1)
            else:
                # For binary classification
                predictions_prob = predictions_prob.flatten()
                predictions = (predictions_prob > 0.5).astype(int)

            # Prepare response
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, predictions_prob)):
                results.append({
                    'prediction': int(pred),
                    'probability': float(prob),
                    'confidence': f"{prob*100:.2f}%"
                })

            return results, 200
        else:
            return {"error": "Prediction failed. Check logs for details."}, 500

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}, 500

# Load artifacts at startup
artifacts = load_artifacts()

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html', health_status="OK" if artifacts is not None else "ERROR")

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions."""
    if artifacts is None:
        return jsonify({"error": "Model artifacts not loaded"}), 500
        
    try:
        # Get JSON data from the request
        input_data = request.get_json(force=True)
        if not input_data:
            return jsonify({"error": "No data provided"}), 400
            
        # Convert to JSON string for processing
        input_json = json.dumps(input_data)
        
        # Make prediction
        results, status_code = make_prediction(input_json, artifacts)
        return jsonify(results), status_code
        
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    if artifacts is None:
        return jsonify({"status": "error", "message": "Model artifacts not loaded"}), 500
    else:
        return jsonify({"status": "ok", "message": "Service is healthy"})

@app.route('/info')
def info():
    """Model information endpoint."""
    if artifacts is None:
        return jsonify({"error": "Model artifacts not loaded"}), 500
    else:
        info = {
            "model_type": artifacts['model_type'],
            "feature_info": artifacts['feature_info'],
        }
        return jsonify(info)

@app.route('/ui/predict', methods=['GET', 'POST'])
def ui_predict():
    """Web interface for making predictions."""
    result = None
    error = None
    
    if request.method == 'POST':
        if artifacts is None:
            error = "Model artifacts not loaded."
        else:
            input_data = request.form.get('input_data')
            if not input_data:
                error = "No data provided."
            else:
                results, status_code = make_prediction(input_data, artifacts)
                if status_code == 200:
                    result = results
                else:
                    error = results.get("error", "An error occurred")
    
    return render_template('predict.html', result=result, error=error)

@app.route('/ui/model_info')
def ui_model_info():
    """Web interface for viewing model information."""
    if artifacts is None or 'model' not in artifacts:
        return render_template('model_info.html', error="Model not loaded.")
    
    model_info = {
        "model_type": artifacts['model_type']
    }
    
    # Get additional model info
    try:
        model = artifacts['model']
        model_type = artifacts['model_type']
        
        if model_type == 'keras_h5':
            import io
            stream = io.StringIO()
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            model_info["summary"] = stream.getvalue()
        else:
            # For SavedModel, show what we can
            if hasattr(model, 'signatures'):
                model_info["signatures"] = list(model.signatures.keys())
            model_info["attributes"] = [attr for attr in dir(model) if not attr.startswith('_')]
    except Exception as e:
        model_info["error"] = f"Error getting model details: {str(e)}"
    
    return render_template('model_info.html', model_info=model_info)

@app.route('/ui/feature_info')
def ui_feature_info():
    """Web interface for viewing feature information."""
    if artifacts is None or 'feature_info' not in artifacts:
        return render_template('feature_info.html', error="Feature info not loaded.")
    
    return render_template('feature_info.html', feature_info=artifacts['feature_info'])

if __name__ == '__main__':
    app.run(debug=True)