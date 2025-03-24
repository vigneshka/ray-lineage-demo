"""
Ray Serve Module for Ray + Iceberg + OpenLineage Demo
This module provides Ray Serve deployment capabilities for hosting models.
"""

import os
import json
import pickle
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import ray
from ray import serve
from starlette.requests import Request


class ModelPredictor:
    """Ray Serve model predictor for inference."""
    
    def __init__(self, 
                model_path: str,
                metadata_path: Optional[str] = None,
                enable_transform: bool = True,
                feature_columns: Optional[List[str]] = None,
                return_probabilities: bool = True):
        """Initialize model predictor.
        
        Args:
            model_path: Path to model file
            metadata_path: Path to model metadata file (optional)
            enable_transform: Whether to apply preprocessing transformations
            feature_columns: List of feature columns to use (if None, use all except target)
            return_probabilities: Whether to return probabilities for classification
        """
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.enable_transform = enable_transform
        self.feature_columns = feature_columns
        self.return_probabilities = return_probabilities
        
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        # Load metadata if available
        self.metadata = {}
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                
            # Use feature columns from metadata if not specified
            if self.feature_columns is None and "feature_columns" in self.metadata:
                self.feature_columns = self.metadata.get("feature_columns")
                
        # Determine if model is a pipeline with preprocessor
        self.has_preprocessor = hasattr(self.model, 'named_steps') and 'preprocessor' in self.model.named_steps
        
        if not self.has_preprocessor:
            print("Warning: Model does not have a preprocessor. Input data must be preprocessed.")
            
        # Initialize request counter and timing for metrics
        self.request_count = 0
        self.total_inference_time = 0
    
    def predict(self, data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Make prediction with model.
        
        Args:
            data: Input data (can be dict, DataFrame, or list of dicts)
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError("Input data must be a dict, list of dicts, or DataFrame")
            
        # Filter columns if specified
        if self.feature_columns is not None:
            # Check for missing columns
            missing_columns = [col for col in self.feature_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required feature columns: {missing_columns}")
                
            input_data = df[self.feature_columns]
        else:
            input_data = df
            
        # Make prediction
        if hasattr(self.model, "predict_proba") and self.return_probabilities:
            try:
                y_proba = self.model.predict_proba(input_data)
                
                # For binary classification, return probability of positive class
                if y_proba.shape[1] == 2:
                    probabilities = y_proba[:, 1].tolist()
                else:
                    probabilities = y_proba.tolist()
                    
                predictions = self.model.predict(input_data).tolist()
                
                result = {
                    "predictions": predictions,
                    "probabilities": probabilities
                }
            except Exception as e:
                result = {
                    "error": f"Error making probability prediction: {str(e)}"
                }
        else:
            try:
                predictions = self.model.predict(input_data).tolist()
                result = {
                    "predictions": predictions
                }
            except Exception as e:
                result = {
                    "error": f"Error making prediction: {str(e)}"
                }
                
        # Add metadata
        end_time = time.time()
        inference_time = end_time - start_time
        
        self.request_count += 1
        self.total_inference_time += inference_time
        
        result["metadata"] = {
            "model_info": {
                "model_path": self.model_path,
                "model_name": self.metadata.get("model_name", "unknown"),
                "model_version": self.metadata.get("model_version", "unknown")
            },
            "inference_time_ms": round(inference_time * 1000, 2),
            "instance_metrics": {
                "request_count": self.request_count,
                "avg_inference_time_ms": round((self.total_inference_time / self.request_count) * 1000, 2)
            }
        }
        
        return result


@serve.deployment
class ModelInferenceService:
    """Ray Serve deployment for model inference."""
    
    def __init__(self, 
                model_path: str,
                metadata_path: Optional[str] = None,
                feature_columns: Optional[List[str]] = None,
                return_probabilities: bool = True):
        """Initialize inference service.
        
        Args:
            model_path: Path to model file
            metadata_path: Path to model metadata file
            feature_columns: List of feature columns to use
            return_probabilities: Whether to return probabilities
        """
        self.predictor = ModelPredictor(
            model_path=model_path,
            metadata_path=metadata_path,
            feature_columns=feature_columns,
            return_probabilities=return_probabilities
        )
        
        # Load metadata
        self.metadata = {}
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                
        # Service info
        self.start_time = time.time()
        
    async def __call__(self, request: Request):
        """Handle inference request.
        
        Args:
            request: Starlette request object
            
        Returns:
            Inference results
        """
        try:
            # Parse request data directly from Starlette request
            data = await request.json()
            
            # Make prediction
            result = self.predictor.predict(data)
            
            return result
        except Exception as e:
            return {
                "error": str(e)
            }
    
    async def health(self, _):
        """Health check endpoint.
        
        Returns:
            Health status
        """
        uptime = time.time() - self.start_time
        
        return {
            "status": "healthy",
            "uptime_seconds": round(uptime, 2),
            "model_info": {
                "model_name": self.metadata.get("model_name", "unknown"),
                "model_version": self.metadata.get("model_version", "unknown")
            },
            "request_count": self.predictor.request_count
        }


class RayServeDeployer:
    """Class for deploying models to Ray Serve."""
    
    def __init__(self, serve_host: str = "http://127.0.0.1:8000"):
        """Initialize Ray Serve deployer.
        
        Args:
            serve_host: Ray Serve host URL
        """
        self.serve_host = serve_host
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        # Initialize Ray Serve if not already started
        # In newer Ray versions, `is_started()` method is no longer available
        # We'll start Serve and catch any exceptions if it's already running
        try:
            serve.start(detached=True)
        except ValueError as e:
            # If Serve is already running, a ValueError will be raised
            if "Ray Serve has already been started" not in str(e):
                raise
    
    def deploy_model(self, 
                    model_path: str,
                    metadata_path: Optional[str] = None,
                    deployment_name: Optional[str] = None,
                    route_prefix: Optional[str] = None,
                    num_replicas: int = 1,
                    feature_columns: Optional[List[str]] = None,
                    return_probabilities: bool = True) -> Dict[str, Any]:
        """Deploy model to Ray Serve.
        
        Args:
            model_path: Path to model file
            metadata_path: Path to model metadata file
            deployment_name: Name for deployment (default: derived from model path)
            route_prefix: Route prefix for deployment (default: same as deployment name)
            num_replicas: Number of replicas
            feature_columns: List of feature columns to use
            return_probabilities: Whether to return probabilities
            
        Returns:
            Deployment info
        """
        # Load metadata if available to get model name
        metadata = {}
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
        # Determine deployment name if not provided
        if deployment_name is None:
            if "model_name" in metadata:
                deployment_name = f"{metadata['model_name']}_{metadata.get('model_version', 'latest')}"
            else:
                deployment_name = os.path.basename(model_path).split('.')[0]
        
        # Determine route prefix if not provided
        if route_prefix is None:
            route_prefix = f"/{deployment_name}"
            
        # Ensure route prefix starts with /
        if not route_prefix.startswith('/'):
            route_prefix = f"/{route_prefix}"
            
        # Create deployment
        deployment = ModelInferenceService.options(
            name=deployment_name,
            num_replicas=num_replicas
        ).bind(
            model_path=model_path,
            metadata_path=metadata_path,
            feature_columns=feature_columns,
            return_probabilities=return_probabilities
        )
        
        # Deploy
        serve.run(deployment, route_prefix=route_prefix)
        
        # Create deployment info
        deployment_info = {
            "deployment_name": deployment_name,
            "route_prefix": route_prefix,
            "model_path": model_path,
            "metadata_path": metadata_path,
            "num_replicas": num_replicas,
            "model_info": {
                "model_name": metadata.get("model_name", "unknown"),
                "model_version": metadata.get("model_version", "unknown")
            },
            "endpoint": f"{self.serve_host}{route_prefix}",
            "health_endpoint": f"{self.serve_host}{route_prefix}/health"
        }
        
        return deployment_info
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all Ray Serve deployments.
        
        Returns:
            List of deployment infos
        """
        try:
            # In newer Ray Serve versions, we use serve.list_deployments()
            deployments = serve.list_deployments()
            
            result = []
            for name, config in deployments.items():
                info = {
                    "name": name,
                    "route_prefix": config.get("route_prefix", f"/{name}"),
                    "num_replicas": config.get("num_replicas", 1),
                    "endpoint": f"{self.serve_host}{config.get('route_prefix', f'/{name}')}"
                }
                result.append(info)
                
            return result
        except Exception as e:
            print(f"Warning: Failed to list deployments: {e}")
            return []
    
    def get_deployment(self, name: str) -> Dict[str, Any]:
        """Get Ray Serve deployment by name.
        
        Args:
            name: Deployment name
            
        Returns:
            Deployment info
            
        Raises:
            ValueError: If deployment not found
        """
        try:
            # In newer Ray Serve versions, we use serve.list_deployments()
            deployments = serve.list_deployments()
            
            if name not in deployments:
                raise ValueError(f"Deployment {name} not found")
                
            config = deployments[name]
            
            return {
                "name": name,
                "route_prefix": config.get("route_prefix", f"/{name}"),
                "num_replicas": config.get("num_replicas", 1),
                "endpoint": f"{self.serve_host}{config.get('route_prefix', f'/{name}')}"
            }
        except Exception as e:
            if "not found" in str(e):
                raise ValueError(f"Deployment {name} not found")
            raise
    
    def delete_deployment(self, name: str) -> Dict[str, Any]:
        """Delete Ray Serve deployment.
        
        Args:
            name: Deployment name
            
        Returns:
            Status info
            
        Raises:
            ValueError: If deployment not found
        """
        try:
            # In newer Ray Serve versions, we use serve.delete_deployment()
            serve.delete_deployment(name)
            return {
                "status": "success",
                "message": f"Deployment {name} deleted successfully"
            }
        except Exception as e:
            raise ValueError(f"Failed to delete deployment {name}: {str(e)}")
    
    def update_deployment(self, 
                         name: str,
                         num_replicas: Optional[int] = None) -> Dict[str, Any]:
        """Update Ray Serve deployment.
        
        Args:
            name: Deployment name
            num_replicas: Number of replicas
            
        Returns:
            Updated deployment info
            
        Raises:
            ValueError: If deployment not found
        """
        try:
            # In newer Ray Serve versions, updating is done differently
            if num_replicas is not None:
                serve.update_deployment(name, num_replicas=num_replicas)
                
            # Get updated deployment info
            deployments = serve.list_deployments()
            if name not in deployments:
                raise ValueError(f"Deployment {name} not found after update")
                
            config = deployments[name]
            return {
                "name": name,
                "route_prefix": config.get("route_prefix", f"/{name}"),
                "num_replicas": config.get("num_replicas", 1),
                "endpoint": f"{self.serve_host}{config.get('route_prefix', f'/{name}')}"
            }
        except Exception as e:
            raise ValueError(f"Failed to update deployment {name}: {str(e)}")
