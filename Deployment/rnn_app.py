import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from datetime import datetime
import logging
from typing import Dict, Any, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_artifacts() -> Union[Dict[str, Any], None]:
    """Load model, preprocessor, and feature info artifacts."""
    artifacts = {}
    
    # Load model - handling both SavedModel and H5 formats
    model_path = 'C:/xampp/htdocs/Projects/CAPSTONE/Deployment/rnn_model_noisy'
    model_h5_path = 'C:/xampp/htdocs/Projects/CAPSTONE/Deployment/rnn_model_noisy.h5'
    
    try:
        # First try loading as H5 model if it exists
        if os.path.exists(model_h5_path):
            st.write(f"Found H5 model at: {model_h5_path}")
            try:
                from tensorflow.python.keras.models import load_model
                artifacts['model'] = load_model(model_h5_path, compile=True)
                st.write("H5 model loaded successfully!")
                logger.info(f"Keras H5 model loaded successfully from {model_h5_path}")
                artifacts['model_type'] = 'keras_h5'
            except Exception as e:
                st.error(f"Error loading H5 model: {str(e)}")
                logger.error(f"Error loading H5 model: {str(e)}")
                return None
        
        # If H5 not found, try loading as SavedModel
        elif os.path.exists(model_path):
            st.write(f"Found SavedModel at: {model_path}")
            try:
                # Load as SavedModel
                artifacts['model'] = tf.saved_model.load(model_path)
                st.write("SavedModel loaded successfully!")
                logger.info(f"SavedModel loaded successfully from {model_path}")
                artifacts['model_type'] = 'saved_model'
            except Exception as e:
                st.error(f"Error loading SavedModel: {str(e)}")
                logger.error(f"Error loading SavedModel: {str(e)}")
                return None
        else:
            st.error(f"No model found at {model_path} or {model_h5_path}")
            return None
    except Exception as e:
        st.error(f"Error in model loading process: {str(e)}")
        logger.error(f"Error in model loading process: {str(e)}")
        return None
        
    # Load preprocessor
    preprocessor_path = 'C:/xampp/htdocs/Projects/CAPSTONE/Deployment/preprocessor_noisy.pkl'
    try:
        st.write(f"Checking if preprocessor file exists at: {preprocessor_path}")
        if os.path.exists(preprocessor_path):
            st.write("Preprocessor file exists, attempting to load...")
            artifacts['preprocessor'] = joblib.load(preprocessor_path)
            st.write("Preprocessor loaded successfully!")
        else:
            st.error(f"Preprocessor file not found at {preprocessor_path}")
            return None
    except Exception as e:
        st.error(f"Error loading preprocessor: {str(e)}")
        return None
        
    # Load feature info
    feature_info_path = 'C:/xampp/htdocs/Projects/CAPSTONE/Deployment/feature_info_noisy.pkl'
    try:
        st.write(f"Checking if feature info file exists at: {feature_info_path}")
        if os.path.exists(feature_info_path):
            st.write("Feature info file exists, attempting to load...")
            artifacts['feature_info'] = joblib.load(feature_info_path)
            st.write("Feature info loaded successfully!")
        else:
            st.error(f"Feature info file not found at {feature_info_path}")
            return None
    except Exception as e:
        st.error(f"Error loading feature info: {str(e)}")
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
            st.write(f"Second prediction method failed: {e2}")
            # Last attempt - try to find a function called predict
            try:
                if hasattr(model, 'predict'):
                    return model.predict(data)
                else:
                    raise AttributeError("Model has no predict method and other attempts failed")
            except Exception as e3:
                st.error(f"All prediction methods failed: {e3}")
                return None

artifacts = load_artifacts()

# Streamlit UI
st.title("RNN Model Prediction App")

# Health Check
if artifacts is None:
    st.error("Model, preprocessor, or feature info not loaded.")
else:
    st.success("All components loaded successfully.")

# Input Data
st.header("Input Data")
input_data = st.text_area("Enter JSON data (single record or array):", height=200)

if st.button("Predict"):
    if not input_data:
        st.error("No data provided.")
    else:
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
                st.error(f"Missing required features: {missing_features}")
            else:
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

                    # Display results
                    st.success("Predictions made successfully!")
                    st.json(results)
                else:
                    st.error("Prediction failed. Check logs for details.")

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

# Model Info
if st.button("Get Model Info"):
    if artifacts is None or 'model' not in artifacts:
        st.error("Model not loaded.")
    else:
        try:
            model = artifacts['model']
            model_type = artifacts['model_type']
            
            if model_type == 'keras_h5':
                import io
                stream = io.StringIO()
                model.summary(print_fn=lambda x: stream.write(x + '\n'))
                summary_str = stream.getvalue()
                st.text_area("Model Summary", summary_str, height=300)
            else:
                # For SavedModel, show what we can
                st.write("Model Type:", model_type)
                st.write("Model Info:")
                if hasattr(model, 'signatures'):
                    st.write("Available signatures:", list(model.signatures.keys()))
                st.write("Model attributes:", [attr for attr in dir(model) if not attr.startswith('_')])
        except Exception as e:
            st.error(f"Error getting model info: {str(e)}")

# Feature Info
if st.button("Get Feature Info"):
    if artifacts is None or 'feature_info' not in artifacts:
        st.error("Feature info not loaded.")
    else:
        try:
            feature_info = artifacts['feature_info']
            st.json(feature_info)
        except Exception as e:
            st.error(f"Error getting feature info: {str(e)}")

if __name__ == '__main__':
    st.write("Streamlit app is running...")