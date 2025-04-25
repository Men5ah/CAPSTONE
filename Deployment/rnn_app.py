# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os
# import tensorflow as tf
# from datetime import datetime
# import logging
# from typing import Dict, Any, Union

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# def load_artifacts() -> Union[Dict[str, Any], None]:
#     """Load model, preprocessor, and feature info artifacts."""
#     artifacts = {}
    
#     # Load model - handling both SavedModel and H5 formats
#     model_path = 'C:/xampp/htdocs/Projects/CAPSTONE/Deployment/rnn_model_noisy'
#     model_h5_path = 'C:/xampp/htdocs/Projects/CAPSTONE/Deployment/rnn_model_noisy.h5'
    
#     try:
#         # First try loading as H5 model if it exists
#         if os.path.exists(model_h5_path):
#             st.write(f"Found H5 model at: {model_h5_path}")
#             try:
#                 from tensorflow.python.keras.models import load_model
#                 artifacts['model'] = load_model(model_h5_path, compile=True)
#                 st.write("H5 model loaded successfully!")
#                 logger.info(f"Keras H5 model loaded successfully from {model_h5_path}")
#                 artifacts['model_type'] = 'keras_h5'
#             except Exception as e:
#                 st.error(f"Error loading H5 model: {str(e)}")
#                 logger.error(f"Error loading H5 model: {str(e)}")
#                 return None
        
#         # If H5 not found, try loading as SavedModel
#         elif os.path.exists(model_path):
#             st.write(f"Found SavedModel at: {model_path}")
#             try:
#                 # Load as SavedModel
#                 artifacts['model'] = tf.saved_model.load(model_path)
#                 st.write("SavedModel loaded successfully!")
#                 logger.info(f"SavedModel loaded successfully from {model_path}")
#                 artifacts['model_type'] = 'saved_model'
#             except Exception as e:
#                 st.error(f"Error loading SavedModel: {str(e)}")
#                 logger.error(f"Error loading SavedModel: {str(e)}")
#                 return None
#         else:
#             st.error(f"No model found at {model_path} or {model_h5_path}")
#             return None
#     except Exception as e:
#         st.error(f"Error in model loading process: {str(e)}")
#         logger.error(f"Error in model loading process: {str(e)}")
#         return None
        
#     # Load preprocessor
#     preprocessor_path = 'C:/xampp/htdocs/Projects/CAPSTONE/Deployment/preprocessor_noisy.pkl'
#     try:
#         st.write(f"Checking if preprocessor file exists at: {preprocessor_path}")
#         if os.path.exists(preprocessor_path):
#             st.write("Preprocessor file exists, attempting to load...")
#             artifacts['preprocessor'] = joblib.load(preprocessor_path)
#             st.write("Preprocessor loaded successfully!")
#         else:
#             st.error(f"Preprocessor file not found at {preprocessor_path}")
#             return None
#     except Exception as e:
#         st.error(f"Error loading preprocessor: {str(e)}")
#         return None
        
#     # Load feature info
#     feature_info_path = 'C:/xampp/htdocs/Projects/CAPSTONE/Deployment/feature_info_noisy.pkl'
#     try:
#         st.write(f"Checking if feature info file exists at: {feature_info_path}")
#         if os.path.exists(feature_info_path):
#             st.write("Feature info file exists, attempting to load...")
#             artifacts['feature_info'] = joblib.load(feature_info_path)
#             st.write("Feature info loaded successfully!")
#         else:
#             st.error(f"Feature info file not found at {feature_info_path}")
#             return None
#     except Exception as e:
#         st.error(f"Error loading feature info: {str(e)}")
#         return None

#     return artifacts

# def predict_with_model(model, data, model_type):
#     """Handle prediction for different model types."""
#     if model_type == 'keras_h5':
#         return model.predict(data)
#     elif model_type == 'saved_model':
#         # For SavedModel format
#         try:
#             # Try using the serving signature
#             infer = model.signatures['serving_default']
#             input_name = list(infer.structured_input_signature[1].keys())[0]
#             result = infer(**{input_name: tf.convert_to_tensor(data, dtype=tf.float32)})
#             output_name = list(result.keys())[0]
#             return result[output_name].numpy()
#         except Exception as e2:
#             st.write(f"Second prediction method failed: {e2}")
#             # Last attempt - try to find a function called predict
#             try:
#                 if hasattr(model, 'predict'):
#                     return model.predict(data)
#                 else:
#                     raise AttributeError("Model has no predict method and other attempts failed")
#             except Exception as e3:
#                 st.error(f"All prediction methods failed: {e3}")
#                 return None

# artifacts = load_artifacts()

# # Streamlit UI
# st.title("RNN Model Prediction App")

# # Health Check
# if artifacts is None:
#     st.error("Model, preprocessor, or feature info not loaded.")
# else:
#     st.success("All components loaded successfully.")

# # Input Data
# st.header("Input Data")
# input_data = st.text_area("Enter JSON data (single record or array):", height=200)

# if st.button("Predict"):
#     if not input_data:
#         st.error("No data provided.")
#     else:
#         try:
#             # Parse input data
#             content = pd.read_json(input_data)

#             # Retrieve numerical and categorical features from feature info
#             numerical_features = artifacts['feature_info'].get('numerical_features', [])
#             categorical_features = artifacts['feature_info'].get('categorical_features', [])

#             # Combine numerical and categorical features for validation
#             required_features = numerical_features + categorical_features

#             # Validate required features
#             missing_features = set(required_features) - set(content.columns)
#             if missing_features:
#                 st.error(f"Missing required features: {missing_features}")
#             else:
#                 # Preprocess the input data
#                 processed_data = artifacts['preprocessor'].transform(content)

#                 # Reshape for RNN input
#                 processed_data_reshaped = np.expand_dims(processed_data, axis=1)

#                 # Make predictions using appropriate method
#                 predictions_prob = predict_with_model(
#                     artifacts['model'], 
#                     processed_data_reshaped, 
#                     artifacts['model_type']
#                 )
                
#                 if predictions_prob is not None:
#                     # Make sure predictions are in the right format
#                     if len(predictions_prob.shape) > 1 and predictions_prob.shape[1] > 1:
#                         # For multi-class output, get the class with highest probability
#                         predictions = np.argmax(predictions_prob, axis=1)
#                     else:
#                         # For binary classification
#                         predictions_prob = predictions_prob.flatten()
#                         predictions = (predictions_prob > 0.5).astype(int)

#                     # Prepare response
#                     results = []
#                     for i, (pred, prob) in enumerate(zip(predictions, predictions_prob)):
#                         results.append({
#                             'prediction': int(pred),
#                             'probability': float(prob),
#                             'confidence': f"{prob*100:.2f}%"
#                         })

#                     # Display results
#                     st.success("Predictions made successfully!")
#                     st.json(results)
#                 else:
#                     st.error("Prediction failed. Check logs for details.")

#         except Exception as e:
#             st.error(f"Error making prediction: {str(e)}")
#             import traceback
#             st.error(traceback.format_exc())

# # Model Info
# if st.button("Get Model Info"):
#     if artifacts is None or 'model' not in artifacts:
#         st.error("Model not loaded.")
#     else:
#         try:
#             model = artifacts['model']
#             model_type = artifacts['model_type']
            
#             if model_type == 'keras_h5':
#                 import io
#                 stream = io.StringIO()
#                 model.summary(print_fn=lambda x: stream.write(x + '\n'))
#                 summary_str = stream.getvalue()
#                 st.text_area("Model Summary", summary_str, height=300)
#             else:
#                 # For SavedModel, show what we can
#                 st.write("Model Type:", model_type)
#                 st.write("Model Info:")
#                 if hasattr(model, 'signatures'):
#                     st.write("Available signatures:", list(model.signatures.keys()))
#                 st.write("Model attributes:", [attr for attr in dir(model) if not attr.startswith('_')])
#         except Exception as e:
#             st.error(f"Error getting model info: {str(e)}")

# # Feature Info
# if st.button("Get Feature Info"):
#     if artifacts is None or 'feature_info' not in artifacts:
#         st.error("Feature info not loaded.")
#     else:
#         try:
#             feature_info = artifacts['feature_info']
#             st.json(feature_info)
#         except Exception as e:
#             st.error(f"Error getting feature info: {str(e)}")

# if __name__ == '__main__':
#     st.write("Streamlit app is running...")







#---------------------------------------------------------------------------------



# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os
# import tensorflow as tf
# from datetime import datetime
# import logging
# import json
# from typing import Dict, Any, Union

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# def load_artifacts() -> Union[Dict[str, Any], None]:
#     """Load model, preprocessor, and feature info artifacts."""
#     artifacts = {}
    
#     # Load model - handling both SavedModel and H5 formats
#     model_path = 'Deployment/rnn_model_noisy'
#     model_h5_path = 'Deployment/rnn_model_noisy.h5'
    
#     try:
#         # First try loading as H5 model if it exists
#         if os.path.exists(model_h5_path):
#             if not st.session_state.get('api_mode', False):
#                 st.write(f"Found H5 model at: {model_h5_path}")
#             try:
#                 from tensorflow.python.keras.models import load_model
#                 artifacts['model'] = load_model(model_h5_path, compile=True)
#                 if not st.session_state.get('api_mode', False):
#                     st.write("H5 model loaded successfully!")
#                 logger.info(f"Keras H5 model loaded successfully from {model_h5_path}")
#                 artifacts['model_type'] = 'keras_h5'
#             except Exception as e:
#                 if not st.session_state.get('api_mode', False):
#                     st.error(f"Error loading H5 model: {str(e)}")
#                 logger.error(f"Error loading H5 model: {str(e)}")
#                 return None
        
#         # If H5 not found, try loading as SavedModel
#         elif os.path.exists(model_path):
#             if not st.session_state.get('api_mode', False):
#                 st.write(f"Found SavedModel at: {model_path}")
#             try:
#                 # Load as SavedModel
#                 artifacts['model'] = tf.saved_model.load(model_path)
#                 if not st.session_state.get('api_mode', False):
#                     st.write("SavedModel loaded successfully!")
#                 logger.info(f"SavedModel loaded successfully from {model_path}")
#                 artifacts['model_type'] = 'saved_model'
#             except Exception as e:
#                 if not st.session_state.get('api_mode', False):
#                     st.error(f"Error loading SavedModel: {str(e)}")
#                 logger.error(f"Error loading SavedModel: {str(e)}")
#                 return None
#         else:
#             if not st.session_state.get('api_mode', False):
#                 st.error(f"No model found at {model_path} or {model_h5_path}")
#             return None
#     except Exception as e:
#         if not st.session_state.get('api_mode', False):
#             st.error(f"Error in model loading process: {str(e)}")
#         logger.error(f"Error in model loading process: {str(e)}")
#         return None
        
#     # Load preprocessor
#     preprocessor_path = 'Deployment/preprocessor_noisy.pkl'
#     try:
#         if not st.session_state.get('api_mode', False):
#             st.write(f"Checking if preprocessor file exists at: {preprocessor_path}")
#         if os.path.exists(preprocessor_path):
#             if not st.session_state.get('api_mode', False):
#                 st.write("Preprocessor file exists, attempting to load...")
#             artifacts['preprocessor'] = joblib.load(preprocessor_path)
#             if not st.session_state.get('api_mode', False):
#                 st.write("Preprocessor loaded successfully!")
#         else:
#             if not st.session_state.get('api_mode', False):
#                 st.error(f"Preprocessor file not found at {preprocessor_path}")
#             return None
#     except Exception as e:
#         if not st.session_state.get('api_mode', False):
#             st.error(f"Error loading preprocessor: {str(e)}")
#         return None
        
#     # Load feature info
#     feature_info_path = 'Deployment/feature_info_noisy.pkl'
#     try:
#         if not st.session_state.get('api_mode', False):
#             st.write(f"Checking if feature info file exists at: {feature_info_path}")
#         if os.path.exists(feature_info_path):
#             if not st.session_state.get('api_mode', False):
#                 st.write("Feature info file exists, attempting to load...")
#             artifacts['feature_info'] = joblib.load(feature_info_path)
#             if not st.session_state.get('api_mode', False):
#                 st.write("Feature info loaded successfully!")
#         else:
#             if not st.session_state.get('api_mode', False):
#                 st.error(f"Feature info file not found at {feature_info_path}")
#             return None
#     except Exception as e:
#         if not st.session_state.get('api_mode', False):
#             st.error(f"Error loading feature info: {str(e)}")
#         return None

#     return artifacts

# def predict_with_model(model, data, model_type):
#     """Handle prediction for different model types."""
#     if model_type == 'keras_h5':
#         return model.predict(data)
#     elif model_type == 'saved_model':
#         # For SavedModel format
#         try:
#             # Try using the serving signature
#             infer = model.signatures['serving_default']
#             input_name = list(infer.structured_input_signature[1].keys())[0]
#             result = infer(**{input_name: tf.convert_to_tensor(data, dtype=tf.float32)})
#             output_name = list(result.keys())[0]
#             return result[output_name].numpy()
#         except Exception as e2:
#             if not st.session_state.get('api_mode', False):
#                 st.write(f"Second prediction method failed: {e2}")
#             # Last attempt - try to find a function called predict
#             try:
#                 if hasattr(model, 'predict'):
#                     return model.predict(data)
#                 else:
#                     raise AttributeError("Model has no predict method and other attempts failed")
#             except Exception as e3:
#                 if not st.session_state.get('api_mode', False):
#                     st.error(f"All prediction methods failed: {e3}")
#                 return None

# def make_prediction(input_data, artifacts):
#     """Process input data and make prediction."""
#     try:
#         # Parse input data
#         content = pd.read_json(input_data)

#         # Retrieve numerical and categorical features from feature info
#         numerical_features = artifacts['feature_info'].get('numerical_features', [])
#         categorical_features = artifacts['feature_info'].get('categorical_features', [])

#         # Combine numerical and categorical features for validation
#         required_features = numerical_features + categorical_features

#         # Validate required features
#         missing_features = set(required_features) - set(content.columns)
#         if missing_features:
#             return {"error": f"Missing required features: {missing_features}"}, 400
        
#         # Preprocess the input data
#         processed_data = artifacts['preprocessor'].transform(content)

#         # Reshape for RNN input
#         processed_data_reshaped = np.expand_dims(processed_data, axis=1)

#         # Make predictions using appropriate method
#         predictions_prob = predict_with_model(
#             artifacts['model'], 
#             processed_data_reshaped, 
#             artifacts['model_type']
#         )
        
#         if predictions_prob is not None:
#             # Make sure predictions are in the right format
#             if len(predictions_prob.shape) > 1 and predictions_prob.shape[1] > 1:
#                 # For multi-class output, get the class with highest probability
#                 predictions = np.argmax(predictions_prob, axis=1)
#             else:
#                 # For binary classification
#                 predictions_prob = predictions_prob.flatten()
#                 predictions = (predictions_prob > 0.5).astype(int)

#             # Prepare response
#             results = []
#             for i, (pred, prob) in enumerate(zip(predictions, predictions_prob)):
#                 results.append({
#                     'prediction': int(pred),
#                     'probability': float(prob),
#                     'confidence': f"{prob*100:.2f}%"
#                 })

#             return results, 200
#         else:
#             return {"error": "Prediction failed. Check logs for details."}, 500

#     except Exception as e:
#         import traceback
#         return {"error": str(e), "traceback": traceback.format_exc()}, 500

# # Main application
# def main():
#     # Check if this is an API call using the non-deprecated approach
#     query_params = st.query_params
    
#     # API mode detection
#     api_mode = "api" in query_params
#     st.session_state['api_mode'] = api_mode
    
#     # Load artifacts
#     artifacts = load_artifacts()
    
#     if api_mode:  # API mode
#         # API endpoints
#         if "predict" in query_params:
#             if artifacts is None:
#                 st.json({"error": "Model artifacts not loaded"})
#                 return
                
#             # Get data from query parameter or request body
#             if "data" in query_params:
#                 input_data = query_params["data"]
#             else:
#                 # Try to get data from request body (this is a hack since Streamlit doesn't natively support this)
#                 try:
#                     request_json = st._get_widget_states()
#                     if "body" in request_json:
#                         input_data = request_json["body"]
#                     else:
#                         st.json({"error": "No data provided in request"})
#                         return
#                 except:
#                     st.json({"error": "No data provided"})
#                     return
                    
#             # Make prediction
#             result, status_code = make_prediction(input_data, artifacts)
#             st.json(result)
            
#         elif "health" in query_params:
#             if artifacts is None:
#                 st.json({"status": "error", "message": "Model artifacts not loaded"})
#             else:
#                 st.json({"status": "ok", "message": "Service is healthy"})
                
#         elif "info" in query_params:
#             if artifacts is None:
#                 st.json({"error": "Model artifacts not loaded"})
#             else:
#                 info = {
#                     "model_type": artifacts['model_type'],
#                     "feature_info": artifacts['feature_info'],
#                 }
#                 st.json(info)
                
#         else:
#             # API documentation
#             st.json({
#                 "message": "RNN Model Prediction API",
#                 "endpoints": {
#                     "/": "API documentation",
#                     "/?api&predict&data={json_data}": "Make prediction with JSON data",
#                     "/?api&health": "Health check",
#                     "/?api&info": "Model information"
#                 }
#             })
            
#     else:  # UI mode
#         # Streamlit UI
#         st.title("RNN Model Prediction App")

#         # Health Check
#         if artifacts is None:
#             st.error("Model, preprocessor, or feature info not loaded.")
#         else:
#             st.success("All components loaded successfully.")

#         # Input Data
#         st.header("Input Data")
#         input_data = st.text_area("Enter JSON data (single record or array):", height=200)

#         if st.button("Predict"):
#             if not input_data:
#                 st.error("No data provided.")
#             else:
#                 results, status_code = make_prediction(input_data, artifacts)
#                 if status_code == 200:
#                     st.success("Predictions made successfully!")
#                     st.json(results)
#                 else:
#                     st.error(results.get("error", "An error occurred"))
#                     if "traceback" in results:
#                         st.error(results["traceback"])

#         # API Usage Info
#         with st.expander("API Usage Information"):
#             st.markdown("""
#             ### How to use this as an API
            
#             This Streamlit app can also function as a simple API. Here are the available endpoints:
            
#             #### Make Predictions
#             ```
#             GET /?api&predict&data={"col1":1,"col2":2}
#             ```
            
#             #### Health Check
#             ```
#             GET /?api&health
#             ```
            
#             #### Get Model Info
#             ```
#             GET /?api&info
#             ```
            
#             #### Example using curl:
#             ```bash
#             curl -X GET "http://localhost:8501/?api&predict&data={\\\"feature1\\\":[1,2],\\\"feature2\\\":[3,4]}"
#             ```
#             """)

#         # Model Info
#         if st.button("Get Model Info"):
#             if artifacts is None or 'model' not in artifacts:
#                 st.error("Model not loaded.")
#             else:
#                 try:
#                     model = artifacts['model']
#                     model_type = artifacts['model_type']
                    
#                     if model_type == 'keras_h5':
#                         import io
#                         stream = io.StringIO()
#                         model.summary(print_fn=lambda x: stream.write(x + '\n'))
#                         summary_str = stream.getvalue()
#                         st.text_area("Model Summary", summary_str, height=300)
#                     else:
#                         # For SavedModel, show what we can
#                         st.write("Model Type:", model_type)
#                         st.write("Model Info:")
#                         if hasattr(model, 'signatures'):
#                             st.write("Available signatures:", list(model.signatures.keys()))
#                         st.write("Model attributes:", [attr for attr in dir(model) if not attr.startswith('_')])
#                 except Exception as e:
#                     st.error(f"Error getting model info: {str(e)}")

#         # Feature Info
#         if st.button("Get Feature Info"):
#             if artifacts is None or 'feature_info' not in artifacts:
#                 st.error("Feature info not loaded.")
#             else:
#                 try:
#                     feature_info = artifacts['feature_info']
#                     st.json(feature_info)
#                 except Exception as e:
#                     st.error(f"Error getting feature info: {str(e)}")

# if __name__ == '__main__':
#     main()
#     if not st.session_state.get('api_mode', False):
#         st.write("Streamlit app is running...")

























import streamlit as st
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

# Add CORS headers to responses
def add_cors_headers():
    st.set_page_config(
        page_title="RNN Model Prediction App",
        page_icon="🤖",
        layout="wide"
    )
    
    # Hack to add CORS headers to Streamlit responses
    from streamlit.web.server.websocket_headers import _get_websocket_headers
    headers = _get_websocket_headers()
    if headers is not None:
        headers['Access-Control-Allow-Origin'] = '*'
        headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        headers['Access-Control-Allow-Headers'] = 'Content-Type'

def load_artifacts() -> Union[Dict[str, Any], None]:
    """Load model, preprocessor, and feature info artifacts."""
    artifacts = {}
    
    # Load model - handling both SavedModel and H5 formats
    model_path = 'Deployment/rnn_model_noisy'
    model_h5_path = 'Deployment/rnn_model_noisy.h5'
    
    try:
        # First try loading as H5 model if it exists
        if os.path.exists(model_h5_path):
            if not st.session_state.get('api_mode', False):
                st.write(f"Found H5 model at: {model_h5_path}")
            try:
                from tensorflow.python.keras.models import load_model
                artifacts['model'] = load_model(model_h5_path, compile=True)
                if not st.session_state.get('api_mode', False):
                    st.write("H5 model loaded successfully!")
                logger.info(f"Keras H5 model loaded successfully from {model_h5_path}")
                artifacts['model_type'] = 'keras_h5'
            except Exception as e:
                if not st.session_state.get('api_mode', False):
                    st.error(f"Error loading H5 model: {str(e)}")
                logger.error(f"Error loading H5 model: {str(e)}")
                return None
        
        # If H5 not found, try loading as SavedModel
        elif os.path.exists(model_path):
            if not st.session_state.get('api_mode', False):
                st.write(f"Found SavedModel at: {model_path}")
            try:
                # Load as SavedModel
                artifacts['model'] = tf.saved_model.load(model_path)
                if not st.session_state.get('api_mode', False):
                    st.write("SavedModel loaded successfully!")
                logger.info(f"SavedModel loaded successfully from {model_path}")
                artifacts['model_type'] = 'saved_model'
            except Exception as e:
                if not st.session_state.get('api_mode', False):
                    st.error(f"Error loading SavedModel: {str(e)}")
                logger.error(f"Error loading SavedModel: {str(e)}")
                return None
        else:
            if not st.session_state.get('api_mode', False):
                st.error(f"No model found at {model_path} or {model_h5_path}")
            return None
    except Exception as e:
        if not st.session_state.get('api_mode', False):
            st.error(f"Error in model loading process: {str(e)}")
        logger.error(f"Error in model loading process: {str(e)}")
        return None
        
    # Load preprocessor
    preprocessor_path = 'Deployment/preprocessor_noisy.pkl'
    try:
        if not st.session_state.get('api_mode', False):
            st.write(f"Checking if preprocessor file exists at: {preprocessor_path}")
        if os.path.exists(preprocessor_path):
            if not st.session_state.get('api_mode', False):
                st.write("Preprocessor file exists, attempting to load...")
            artifacts['preprocessor'] = joblib.load(preprocessor_path)
            if not st.session_state.get('api_mode', False):
                st.write("Preprocessor loaded successfully!")
        else:
            if not st.session_state.get('api_mode', False):
                st.error(f"Preprocessor file not found at {preprocessor_path}")
            return None
    except Exception as e:
        if not st.session_state.get('api_mode', False):
            st.error(f"Error loading preprocessor: {str(e)}")
        return None
        
    # Load feature info
    feature_info_path = 'Deployment/feature_info_noisy.pkl'
    try:
        if not st.session_state.get('api_mode', False):
            st.write(f"Checking if feature info file exists at: {feature_info_path}")
        if os.path.exists(feature_info_path):
            if not st.session_state.get('api_mode', False):
                st.write("Feature info file exists, attempting to load...")
            artifacts['feature_info'] = joblib.load(feature_info_path)
            if not st.session_state.get('api_mode', False):
                st.write("Feature info loaded successfully!")
        else:
            if not st.session_state.get('api_mode', False):
                st.error(f"Feature info file not found at {feature_info_path}")
            return None
    except Exception as e:
        if not st.session_state.get('api_mode', False):
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
            if not st.session_state.get('api_mode', False):
                st.write(f"Second prediction method failed: {e2}")
            # Last attempt - try to find a function called predict
            try:
                if hasattr(model, 'predict'):
                    return model.predict(data)
                else:
                    raise AttributeError("Model has no predict method and other attempts failed")
            except Exception as e3:
                if not st.session_state.get('api_mode', False):
                    st.error(f"All prediction methods failed: {e3}")
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

# Main application
def main():
    # Add CORS headers
    add_cors_headers()
    
    # Check if this is an API call using the non-deprecated approach
    query_params = st.query_params
    
    # API mode detection
    api_mode = "api" in query_params
    st.session_state['api_mode'] = api_mode
    
    # Load artifacts
    artifacts = load_artifacts()
    
    if api_mode:  # API mode
        # API endpoints
        if "predict" in query_params:
            if artifacts is None:
                st.json({"error": "Model artifacts not loaded"})
                return
                
            # Get data from query parameter or request body
            if "data" in query_params:
                input_data = query_params["data"]
            else:
                # Try to get data from request body (this is a hack since Streamlit doesn't natively support this)
                try:
                    request_json = st._get_widget_states()
                    if "body" in request_json:
                        input_data = request_json["body"]
                    else:
                        st.json({"error": "No data provided in request"})
                        return
                except:
                    st.json({"error": "No data provided"})
                    return
                    
            # Make prediction
            result, status_code = make_prediction(input_data, artifacts)
            st.json(result)
            
        elif "health" in query_params:
            if artifacts is None:
                st.json({"status": "error", "message": "Model artifacts not loaded"})
            else:
                st.json({"status": "ok", "message": "Service is healthy"})
                
        elif "info" in query_params:
            if artifacts is None:
                st.json({"error": "Model artifacts not loaded"})
            else:
                info = {
                    "model_type": artifacts['model_type'],
                    "feature_info": artifacts['feature_info'],
                }
                st.json(info)
                
        else:
            # API documentation
            st.json({
                "message": "RNN Model Prediction API",
                "endpoints": {
                    "/": "API documentation",
                    "/?api&predict&data={json_data}": "Make prediction with JSON data",
                    "/?api&health": "Health check",
                    "/?api&info": "Model information"
                },
                "note": "When calling from JavaScript, make sure to handle CORS appropriately."
            })
            
    else:  # UI mode
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
                results, status_code = make_prediction(input_data, artifacts)
                if status_code == 200:
                    st.success("Predictions made successfully!")
                    st.json(results)
                else:
                    st.error(results.get("error", "An error occurred"))
                    if "traceback" in results:
                        st.error(results["traceback"])

        # API Usage Info
        with st.expander("API Usage Information"):
            st.markdown("""
            ### How to use this as an API
            
            This Streamlit app can also function as a simple API. Here are the available endpoints:
            
            #### Make Predictions
            ```
            GET /?api&predict&data={"col1":1,"col2":2}
            ```
            
            #### Health Check
            ```
            GET /?api&health
            ```
            
            #### Get Model Info
            ```
            GET /?api&info
            ```
            
            #### Example using curl:
            ```bash
            curl -X GET "http://localhost:8501/?api&predict&data={\\\"feature1\\\":[1,2],\\\"feature2\\\":[3,4]}"
            ```
            
            #### Example using JavaScript:
            ```javascript
            fetch('http://localhost:8501/?api&predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({feature1: [1,2], feature2: [3,4]})
            })
            .then(response => response.json())
            .then(data => console.log(data));
            ```
            """)

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
    main()
    if not st.session_state.get('api_mode', False):
        st.write("Streamlit app is running...")