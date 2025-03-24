"""
Model Registry Module for Ray + Iceberg + OpenLineage Demo
This module provides model registry capabilities for tracking model versions and deployments.
"""

import os
import json
import pickle
import shutil
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import pandas as pd


class ModelRegistry:
    """Model registry for tracking model versions and deployments."""
    
    def __init__(self, registry_dir: str = "./model_registry"):
        """Initialize model registry.
        
        Args:
            registry_dir: Directory for model registry storage
        """
        self.registry_dir = registry_dir
        self.registry_file = os.path.join(registry_dir, "registry.json")
        
        # Create registry directory if it doesn't exist
        os.makedirs(registry_dir, exist_ok=True)
        
        # Create models and deployments directories if they don't exist
        os.makedirs(os.path.join(registry_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(registry_dir, "deployments"), exist_ok=True)
        
        # Load or create registry
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load registry from file or create new registry."""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "models": {},
                "deployments": {}
            }
            self._save_registry()
    
    def _save_registry(self) -> None:
        """Save registry to file."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, 
                      model_path: str, 
                      metadata_path: str,
                      model_name: Optional[str] = None, 
                      model_version: Optional[str] = None,
                      description: Optional[str] = None,
                      tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Register a model in the registry.
        
        Args:
            model_path: Path to model file (.pkl)
            metadata_path: Path to model metadata (.json)
            model_name: Name to register model under (override from metadata)
            model_version: Version to register model under (override from metadata)
            description: Model description
            tags: Tags for the model
            
        Returns:
            Model info dictionary
        """
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Use provided model_name/version or extract from metadata
        model_name = model_name or metadata.get("model_name", "unknown_model")
        model_version = model_version or metadata.get("model_version", "0.0.0")
        
        # Generate model ID
        model_id = f"{model_name}_{model_version}"
        timestamp = datetime.datetime.now().isoformat()
        
        # Create model directory in registry
        model_dir = os.path.join(self.registry_dir, "models", model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Copy model and metadata to registry
        model_registry_path = os.path.join(model_dir, f"{model_id}.pkl")
        metadata_registry_path = os.path.join(model_dir, f"{model_id}_metadata.json")
        
        shutil.copy2(model_path, model_registry_path)
        shutil.copy2(metadata_path, metadata_registry_path)
        
        # Create model info
        model_info = {
            "model_id": model_id,
            "model_name": model_name,
            "model_version": model_version,
            "description": description or metadata.get("description", ""),
            "tags": tags or [],
            "registered_at": timestamp,
            "path": model_registry_path,
            "metadata_path": metadata_registry_path,
            "metadata": metadata,
            "stage": "Registered",
            "deployments": []
        }
        
        # Update registry
        if model_name not in self.registry["models"]:
            self.registry["models"][model_name] = {}
            
        self.registry["models"][model_name][model_version] = model_info
        self._save_registry()
        
        print(f"Model {model_id} registered successfully")
        return model_info
        
    def get_model(self, 
                 model_name: str, 
                 model_version: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """Get a model from the registry.
        
        Args:
            model_name: Model name
            model_version: Model version (if None, latest version is used)
            
        Returns:
            Tuple of (model, model_info)
            
        Raises:
            ValueError: If model not found
        """
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
            
        versions = self.registry["models"][model_name]
        
        if not versions:
            raise ValueError(f"No versions found for model {model_name}")
            
        if model_version is None:
            # Find latest version
            latest_version = sorted(versions.keys(), key=lambda v: [int(x) for x in v.split('.')])[-1]
            model_version = latest_version
        
        if model_version not in versions:
            raise ValueError(f"Version {model_version} not found for model {model_name}")
            
        model_info = versions[model_version]
        model_path = model_info["path"]
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        return model, model_info
        
    def list_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all models in registry.
        
        Returns:
            Dictionary of model name -> list of model versions
        """
        result = {}
        
        for model_name, versions in self.registry["models"].items():
            result[model_name] = []
            
            for version, info in versions.items():
                result[model_name].append({
                    "model_id": info["model_id"],
                    "model_version": version,
                    "registered_at": info["registered_at"],
                    "stage": info["stage"],
                    "description": info["description"],
                    "tags": info["tags"]
                })
                
        return result
        
    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a model.
        
        Args:
            model_name: Model name
            
        Returns:
            List of model version infos
        """
        if model_name not in self.registry["models"]:
            return []
            
        versions = []
        for version, info in self.registry["models"][model_name].items():
            versions.append({
                "model_id": info["model_id"],
                "model_version": version,
                "registered_at": info["registered_at"],
                "stage": info["stage"],
                "description": info["description"],
                "tags": info["tags"],
                "performance": info["metadata"].get("performance", {})
            })
            
        # Sort by version
        versions.sort(key=lambda x: [int(v) for v in x["model_version"].split('.')])
        
        return versions
        
    def compare_model_versions(self, 
                              model_name: str, 
                              versions: Optional[List[str]] = None) -> pd.DataFrame:
        """Compare metrics across model versions.
        
        Args:
            model_name: Model name
            versions: List of versions to compare (if None, all versions are compared)
            
        Returns:
            DataFrame with version comparison
        """
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
            
        model_versions = self.registry["models"][model_name]
        
        if versions is None:
            versions = list(model_versions.keys())
            
        comparison_data = []
        
        for version in versions:
            if version not in model_versions:
                print(f"Warning: Version {version} not found for model {model_name}")
                continue
                
            model_info = model_versions[version]
            metadata = model_info["metadata"]
            
            # Extract metrics
            metrics = {}
            
            # Add basic info
            metrics["version"] = version
            metrics["registered_at"] = model_info["registered_at"]
            metrics["stage"] = model_info["stage"]
            
            # Add training info
            for k in ["training_samples", "test_samples", "training_split", "random_state"]:
                if k in metadata:
                    metrics[k] = metadata[k]
            
            # Add hyperparameters as hp_{name}
            if "hyperparameters" in metadata:
                for hp_name, hp_value in metadata["hyperparameters"].items():
                    metrics[f"hp_{hp_name}"] = hp_value
            
            # Add performance metrics
            for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                if metric_name in metadata:
                    metrics[metric_name] = metadata[metric_name]
            
            comparison_data.append(metrics)
            
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        
        return df
        
    def create_deployment(self, 
                        model_name: str, 
                        model_version: Optional[str] = None,
                        deployment_name: Optional[str] = None,
                        description: Optional[str] = None,
                        config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a deployment for a model.
        
        Args:
            model_name: Model name
            model_version: Model version (if None, latest version is used)
            deployment_name: Name for deployment (default: model_name)
            description: Deployment description
            config: Deployment configuration
            
        Returns:
            Deployment info dictionary
        """
        # Get model info
        _, model_info = self.get_model(model_name, model_version)
        model_version = model_info["model_version"]
        model_id = model_info["model_id"]
        
        # Generate deployment info
        timestamp = datetime.datetime.now().isoformat()
        deployment_name = deployment_name or model_name
        deployment_id = f"{deployment_name}_{timestamp.replace(':', '-')}"
        
        # Create deployment directory
        deployment_dir = os.path.join(self.registry_dir, "deployments", deployment_id)
        os.makedirs(deployment_dir, exist_ok=True)
        
        # Create symlink to model file in deployment dir
        model_path = model_info["path"]
        model_link = os.path.join(deployment_dir, os.path.basename(model_path))
        os.symlink(os.path.abspath(model_path), model_link)
        
        # Create deployment config file
        config = config or {}
        config_path = os.path.join(deployment_dir, f"{deployment_id}_config.json")
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create deployment info
        deployment_info = {
            "deployment_id": deployment_id,
            "deployment_name": deployment_name,
            "model_name": model_name,
            "model_version": model_version,
            "model_id": model_id,
            "description": description or f"Deployment of {model_id}",
            "created_at": timestamp,
            "status": "Created",
            "config": config,
            "config_path": config_path,
            "model_path": model_link
        }
        
        # Update registry
        self.registry["deployments"][deployment_id] = deployment_info
        
        # Update model info
        self.registry["models"][model_name][model_version]["deployments"].append(deployment_id)
        
        self._save_registry()
        
        print(f"Deployment {deployment_id} created successfully")
        return deployment_info
        
    def get_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment info.
        
        Args:
            deployment_id: Deployment ID
            
        Returns:
            Deployment info dictionary
            
        Raises:
            ValueError: If deployment not found
        """
        if deployment_id not in self.registry["deployments"]:
            raise ValueError(f"Deployment {deployment_id} not found")
            
        return self.registry["deployments"][deployment_id]
        
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments.
        
        Returns:
            List of deployment infos
        """
        deployments = []
        
        for deployment_id, info in self.registry["deployments"].items():
            deployments.append({
                "deployment_id": deployment_id,
                "deployment_name": info["deployment_name"],
                "model_name": info["model_name"],
                "model_version": info["model_version"],
                "created_at": info["created_at"],
                "status": info["status"],
                "description": info["description"]
            })
            
        return deployments
        
    def update_deployment_status(self, deployment_id: str, status: str) -> Dict[str, Any]:
        """Update deployment status.
        
        Args:
            deployment_id: Deployment ID
            status: New status
            
        Returns:
            Updated deployment info
            
        Raises:
            ValueError: If deployment not found
        """
        if deployment_id not in self.registry["deployments"]:
            raise ValueError(f"Deployment {deployment_id} not found")
            
        self.registry["deployments"][deployment_id]["status"] = status
        self._save_registry()
        
        return self.registry["deployments"][deployment_id]
        
    def update_model_stage(self, 
                          model_name: str, 
                          model_version: str,
                          stage: str) -> Dict[str, Any]:
        """Update model stage.
        
        Args:
            model_name: Model name
            model_version: Model version
            stage: New stage
            
        Returns:
            Updated model info
            
        Raises:
            ValueError: If model not found
        """
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model {model_name} not found")
            
        if model_version not in self.registry["models"][model_name]:
            raise ValueError(f"Version {model_version} not found for model {model_name}")
            
        self.registry["models"][model_name][model_version]["stage"] = stage
        self._save_registry()
        
        return self.registry["models"][model_name][model_version]
