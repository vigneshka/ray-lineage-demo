#!/usr/bin/env python3
"""
Model Serving Demonstration for Ray + Iceberg + OpenLineage

This script demonstrates:
1. Deploying multiple versions of models to Ray Serve
2. Making predictions with each model version
3. Comparing predictions between model versions
"""

import os
import ray
import pandas as pd
import numpy as np
import json
import time
import pickle
import requests
from sklearn.pipeline import Pipeline

# Initialize Ray if not already initialized
if not ray.is_initialized():
    ray.init()

# Check if Ray Serve is available (since we had issues with this)
try:
    from ray import serve
    SERVE_AVAILABLE = True
except ImportError:
    SERVE_AVAILABLE = False
    print("⚠️ Ray Serve not available. Will demonstrate model loading and prediction locally.")

# MODEL LOADING AND PREDICTION FUNCTIONS

def load_model(model_version):
    """Load a trained model by version."""
    model_path = f"./models/churn_predictor_{model_version}.pkl"
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def load_metadata(model_version):
    """Load model metadata by version."""
    metadata_path = f"./models/churn_predictor_{model_version}_metadata.json"
    with open(metadata_path, 'r') as f:
        return json.load(f)

def predict(model, data):
    """Make predictions with the model."""
    if isinstance(model, Pipeline):
        # If it's a scikit-learn pipeline, use predict and predict_proba
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]  # Probability of class 1
        return predictions, probabilities
    else:
        # If it's a different model type, just use predict
        return model.predict(data), None

# RAY SERVE DEPLOYMENT FUNCTIONS (if available)

if SERVE_AVAILABLE:
    # Check if Ray Serve is already running
    try:
        serve.connect()
        print("Connected to existing Ray Serve instance")
    except:
        # Start Ray Serve
        serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})
        print("Started Ray Serve on port 8000")

    # Define a basic deployment class
    @serve.deployment
    class ChurnPredictor:
        def __init__(self, model_version):
            self.model_version = model_version
            self.model = load_model(model_version)
            self.metadata = load_metadata(model_version)
            print(f"Loaded model version {model_version}")
            
        async def __call__(self, request):
            try:
                # Parse the request data
                json_data = await request.json()
                
                # Convert to pandas DataFrame
                df = pd.DataFrame([json_data])
                
                # Get feature columns from metadata
                feature_columns = self.metadata["model_info"]["feature_columns"]
                
                # Make sure all required columns are present
                for col in feature_columns:
                    if col not in df.columns:
                        df[col] = 0  # Default value for missing columns
                
                # Get only the required columns in the right order
                df = df[feature_columns]
                
                # Make prediction
                predictions, probabilities = predict(self.model, df)
                
                # Return the prediction result
                result = {
                    "model_version": self.model_version,
                    "predictions": predictions.tolist(),
                    "probabilities": probabilities.tolist() if probabilities is not None else None,
                    "metadata": {
                        "model_info": self.metadata["model_info"]
                    }
                }
                return result
            except Exception as e:
                return {"error": str(e)}

# MAIN DEMONSTRATION CODE

print("=" * 60)
print("Model Serving Demonstration")
print("=" * 60)

# Part 1: Load models locally
print("\n=== Loading Models Locally ===")
model_v1 = load_model("1.0.0")
metadata_v1 = load_metadata("1.0.0")
print(f"Loaded model v1.0.0 ({type(model_v1).__name__})")

try:
    model_v2 = load_model("2.0.0")
    metadata_v2 = load_metadata("2.0.0")
    print(f"Loaded model v2.0.0 ({type(model_v2).__name__})")
    two_models_available = True
except FileNotFoundError:
    print("Model v2.0.0 not found. Please run the versioning_demo.py script first.")
    two_models_available = False

# Part 2: Generate test data
print("\n=== Generating Test Data ===")
test_data = pd.DataFrame({
    'Age': [42, 35, 57, 22, 48],
    'Tenure': [5, 10, 2, 1, 8],
    'ContractType': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year'],
    'MonthlyCharges': [79.85, 65.30, 110.75, 45.25, 98.40],
    'TotalCharges': [399.25, 653.00, 221.50, 45.25, 787.20],
    # Add missing columns
    'HasPhoneService': [True, True, False, True, True],
    'HasInternetService': [True, False, True, False, True]
})
print("Test data generated with 5 sample customers:")
print(test_data)

# Part 3: Make local predictions
print("\n=== Making Local Predictions ===")
pred_v1, prob_v1 = predict(model_v1, test_data)
print("\nPredictions with model v1.0.0:")
for i, (pred, prob) in enumerate(zip(pred_v1, prob_v1)):
    print(f"Customer {i+1}: Churn = {pred} (Probability: {prob:.4f})")

if two_models_available:
    pred_v2, prob_v2 = predict(model_v2, test_data)
    print("\nPredictions with model v2.0.0:")
    for i, (pred, prob) in enumerate(zip(pred_v2, prob_v2)):
        print(f"Customer {i+1}: Churn = {pred} (Probability: {prob:.4f})")
    
    # Compare predictions
    print("\nComparison of predictions:")
    for i in range(len(test_data)):
        v1_result = "Will Churn" if pred_v1[i] else "Won't Churn"
        v2_result = "Will Churn" if pred_v2[i] else "Won't Churn"
        agreement = "✓" if pred_v1[i] == pred_v2[i] else "✗"
        print(f"Customer {i+1}: Model v1: {v1_result} ({prob_v1[i]:.2f}), Model v2: {v2_result} ({prob_v2[i]:.2f}) - Agreement: {agreement}")

# Part 4: Deploy models to Ray Serve (if available)
if SERVE_AVAILABLE:
    print("\n=== Deploying Models to Ray Serve ===")
    
    # Deploy v1.0.0
    predictor_v1 = ChurnPredictor.bind("1.0.0")
    serve.run(predictor_v1, name="predictor_v1", route_prefix="/predict/v1")
    print("Model v1.0.0 deployed at: http://localhost:8000/predict/v1")
    
    if two_models_available:
        # Deploy v2.0.0
        predictor_v2 = ChurnPredictor.bind("2.0.0")
        serve.run(predictor_v2, name="predictor_v2", route_prefix="/predict/v2")
        print("Model v2.0.0 deployed at: http://localhost:8000/predict/v2")
    
    # Part 5: Make predictions through Ray Serve
    print("\n=== Making Predictions through Ray Serve ===")
    
    # Test customer to send to the API
    test_customer = {
        'Age': 42,
        'Tenure': 5,
        'ContractType': 'Month-to-month',
        'MonthlyCharges': 79.85,
        'TotalCharges': 399.25,
        'HasPhoneService': True,
        'HasInternetService': True
    }
    
    print(f"Sending test customer to API endpoints:")
    print(json.dumps(test_customer, indent=2))
    
    try:
        # Send to v1
        response_v1 = requests.post("http://localhost:8000/predict/v1", json=test_customer)
        result_v1 = response_v1.json()
        print("\nPrediction from model v1.0.0 API:")
        print(f"Churn: {result_v1['predictions'][0]}")
        print(f"Probability: {result_v1['probabilities'][0]:.4f}")
        
        if two_models_available:
            # Send to v2
            response_v2 = requests.post("http://localhost:8000/predict/v2", json=test_customer)
            result_v2 = response_v2.json()
            print("\nPrediction from model v2.0.0 API:")
            print(f"Churn: {result_v2['predictions'][0]}")
            print(f"Probability: {result_v2['probabilities'][0]:.4f}")
    except Exception as e:
        print(f"Error making API requests: {e}")

print("\n=== Summary ===")
print("This demonstration showed:")
print("1. Loading different model versions")
print("2. Making predictions with each model version locally")
if SERVE_AVAILABLE:
    print("3. Deploying multiple model versions to Ray Serve")
    print("4. Making predictions through the Ray Serve API")
print("\nThese features enable A/B testing, gradual rollouts, and model version comparison.")

# Clean up resources
if 'serve' in locals() and SERVE_AVAILABLE:
    print("\nShutting down Ray Serve...")
    serve.shutdown()

# Shutdown Ray (only if we started it)
if ray.is_initialized():
    ray.shutdown() 