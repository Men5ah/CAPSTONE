# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Copy requirements file if exists, else install needed packages directly
# Since no requirements.txt provided, install packages directly
RUN pip install --no-cache-dir flask pandas numpy joblib tensorflow

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the port the app runs on
EXPOSE 5001

# Set environment variables for model paths (optional, can be overridden)
ENV RNN_MODEL_PATH=rnn_model_noisy.keras
ENV PREPROCESSOR_PATH=preprocessor_noisy.pkl
ENV FEATURE_INFO_PATH=feature_info_noisy.pkl
ENV PORT=5001

# Run the application
CMD ["python", "rnn.py"]
