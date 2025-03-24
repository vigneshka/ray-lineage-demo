#!/usr/bin/env python3
"""
Simplified Ray + Iceberg + OpenLineage Demo

This script demonstrates:
1. Ray Data used to generate Iceberg datasets with versioning
2. Ray Train to generate versioned models
3. Ray Serve serving the model
4. OpenLineage for tracking lineage in Marquez

Everything runs locally without Docker for Ray, but uses Docker for Marquez.
"""

import os
import json
import time
import argparse
import sys

# Check required dependencies first
def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        "ray", "pandas", "numpy", "tensorflow", 
        "matplotlib", "psutil", "requests"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages: " + ", ".join(missing_packages))
        print("Please install these packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        print("Or install all requirements with:")
        print("pip install -r requirements.txt")
        return False
    
    return True

# Ensure all dependencies are installed before proceeding
if not check_dependencies():
    sys.exit(1)

# Import remaining packages after dependency check
import ray
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import psutil
import getpass
import socket

# Try importing TensorFlow with clear error messaging
try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} is available")
except ImportError:
    print("❌ TensorFlow is required but not installed.")
    print("Please install TensorFlow using:")
    print("pip install tensorflow>=2.10.0")
    print("Or install all requirements with:")
    print("pip install -r requirements.txt")
    sys.exit(1)

# Import components from src modules
from src.data.generators import DataGenerator
from src.data.iceberg_manager import IcebergManager
from src.models.trainer import SklearnModelTrainer
from src.models.registry import ModelRegistry
from src.serving.ray_serve import RayServeDeployer
from src.lineage.openlineage_client import OpenLineageClient, LineageTracker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simplified Ray + Iceberg + OpenLineage Demo")
    
    # General options
    parser.add_argument("--storage-path", type=str, default="./storage",
                        help="Path for storing datasets and models")
    parser.add_argument("--openlineage-url", type=str, 
                        default="http://localhost:5002/api/v1/lineage",
                        help="OpenLineage API URL (should point to Marquez API's lineage endpoint)")
    parser.add_argument("--namespace", type=str, default="ray_iceberg_demo",
                        help="OpenLineage namespace")
    
    # Data generation options
    parser.add_argument("--dataset-rows", type=int, default=10000,
                        help="Number of rows to generate for datasets")
    parser.add_argument("--dataset-version", type=str, default="1.0.0",
                        help="Version of dataset to generate")
    
    # Model training options
    parser.add_argument("--model-type", type=str, default="random_forest",
                        choices=["random_forest", "gradient_boosting", "logistic_regression"],
                        help="Type of model to train")
    parser.add_argument("--model-version", type=str, default="1.0.0",
                        help="Version of model to train")
    
    # Ray configuration
    parser.add_argument("--ray-port", type=int, default=6379,
                        help="Port for Ray head node")
    parser.add_argument("--dashboard-port", type=int, default=8265,
                        help="Port for Ray dashboard")
    parser.add_argument("--serve-port", type=int, default=8000,
                        help="Port for Ray Serve")
    parser.add_argument("--ray-num-cpus", type=int, default=None,
                        help="Number of CPUs to use (default: all available)")
    
    # Marquez options
    parser.add_argument("--marquez-web-url", type=str, default="http://localhost:3000",
                        help="Marquez web UI URL")
    
    return parser.parse_args()


def is_port_in_use(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def find_ray_processes():
    """Find Ray processes."""
    ray_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'username']):
        try:
            # Check if this process is a Ray process owned by current user
            if proc.info['username'] == getpass.getuser():
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if 'ray' in cmdline and ('start' in cmdline or 'gcs_server' in cmdline or 'raylet' in cmdline):
                    ray_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return ray_processes


def setup_ray_cluster(args):
    """Set up Ray cluster."""
    print("\n=== Setting up Ray Cluster ===")
    
    # Check if Ray is already initialized
    if ray.is_initialized():
        print("Ray is already initialized in this process. Shutting down...")
        ray.shutdown()
    
    # Check if Ray is already running
    ray_processes = find_ray_processes()
    if ray_processes:
        print(f"Found {len(ray_processes)} Ray processes already running")
        
        try:
            # Try connecting to existing cluster
            ray.init(address="auto", ignore_reinit_error=True)
            print("✅ Connected to existing Ray cluster")
            
            # Get dashboard URL
            dashboard_url = f"http://127.0.0.1:{args.dashboard_port}"
            print(f"Ray Dashboard: {dashboard_url}")
            
            return True
        except Exception as e:
            print(f"Failed to connect to existing Ray cluster: {e}")
            print("Will start a new Ray cluster...")
    
    # Check if ports are available
    ports_in_use = False
    ports_to_check = [
        (args.ray_port, "Ray head node"),
        (args.dashboard_port, "Ray dashboard"),
        (args.serve_port, "Ray Serve")
    ]
    
    for port, service in ports_to_check:
        if is_port_in_use(port):
            print(f"⚠️ Port {port} ({service}) is already in use.")
            ports_in_use = True
    
    # If any ports are in use, try to connect to an existing cluster
    if ports_in_use:
        print("Some required ports are already in use. Attempting to connect to existing Ray cluster...")
        try:
            ray.init(address="auto", ignore_reinit_error=True)
            print("✅ Connected to existing Ray cluster")
            
            # Get dashboard URL if possible
            try:
                dashboard_url = f"http://127.0.0.1:{args.dashboard_port}"
                print(f"Ray Dashboard: {dashboard_url}")
            except:
                print("Could not determine Ray Dashboard URL")
                
            return True
        except Exception as e:
            print(f"Failed to connect to existing Ray cluster: {e}")
            print("⚠️ Please stop any services using the required ports and try again")
            return False
    
    # Set environment variables for Ray Serve
    os.environ["RAY_SERVE_HTTP_PORT"] = str(args.serve_port)
    os.environ["RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING"] = "1"
    
    print(f"Starting Ray head node on port {args.ray_port}...")
    
    # Build Ray init params
    ray_params = {
        "address": None,  # Start a new cluster
        "include_dashboard": True,
        "dashboard_host": "0.0.0.0",
        "dashboard_port": args.dashboard_port,
    }
    
    if args.ray_num_cpus is not None:
        ray_params["num_cpus"] = args.ray_num_cpus
    
    # Initialize Ray
    try:
        ray.init(**ray_params)
        
        # Get dashboard URL 
        dashboard_url = f"http://127.0.0.1:{args.dashboard_port}"
        
        print("\n----- Ray Cluster Information -----")
        resources = ray.cluster_resources()
        print(f"Available CPUs: {resources.get('CPU', 0)}")
        print(f"Available GPUs: {resources.get('GPU', 0)}")
        print(f"Available memory: {resources.get('memory', 0) / 1e9:.2f} GB")
        print(f"Ray Dashboard: {dashboard_url}")
        print(f"Ray Serve URL: http://localhost:{args.serve_port}")
        print("----------------------------------\n")
        
        return True
    except Exception as e:
        print(f"Failed to initialize Ray: {e}")
        return False


def setup_environment(args):
    """Set up environment for demo."""
    print("\n=== Setting up Environment ===")
    
    # Create storage directories
    os.makedirs(args.storage_path, exist_ok=True)
    os.makedirs(os.path.join(args.storage_path, "datasets"), exist_ok=True)
    os.makedirs(os.path.join(args.storage_path, "models"), exist_ok=True)
    
    # Set up Ray cluster
    ray_setup_success = setup_ray_cluster(args)
    if not ray_setup_success:
        print("Failed to set up Ray cluster")
        return None
    
    # Initialize components
    iceberg_manager = IcebergManager(
        warehouse_path=os.path.join(args.storage_path, "iceberg_warehouse"),
        namespace=args.namespace,
        catalog_name="ray_demo_catalog"
    )
    
    model_registry = ModelRegistry(
        registry_dir=os.path.join(args.storage_path, "model_registry")
    )
    
    # Set up OpenLineage client
    openlineage_client = OpenLineageClient(
        api_url=args.openlineage_url,
        namespace=args.namespace
    )
    
    return {
        "iceberg_manager": iceberg_manager,
        "model_registry": model_registry,
        "openlineage_client": openlineage_client,
        "lineage_tracker": LineageTracker(
            client=openlineage_client,
            job_name="environment_setup",
            job_description="Set up the environment for the Ray Iceberg demo"
        )
    }


def generate_datasets(args, env):
    """Generate datasets using Ray Data and Iceberg."""
    print("\n=== Generating Datasets with Ray Data ===")
    
    iceberg_manager = env["iceberg_manager"]
    
    datasets = {}
    
    # Generate customer churn dataset
    with LineageTracker(
        client=env["openlineage_client"],
        job_name="generate_customer_dataset",
        job_description=f"Generate customer churn dataset v{args.dataset_version}"
    ) as lineage:
        # Generate dataset
        print("Generating customer churn dataset...")
        df, metadata = DataGenerator.generate_customer_dataset(
            n_rows=args.dataset_rows,
            version=args.dataset_version,
            random_seed=42,
            noise_level=0.5,
            as_ray_dataset=False  # Get a pandas DataFrame instead of Ray Dataset
        )
        
        # Track dataset as input to this job (for data generation this is a special case)
        lineage.add_job_facet("generator_config", {
            "n_rows": args.dataset_rows,
            "version": args.dataset_version,
            "random_seed": 42,
            "noise_level": 0.5
        })
        
        # Store dataset in Iceberg
        table_name = "customer_churn"
        
        # Write to Iceberg using native Ray Data method with lineage tracking
        table_metadata = iceberg_manager.create_table(
            table_name=table_name,
            df=df,
            metadata=metadata,
            lineage_tracker=lineage
        )
        
        # Store for other functions to use
        datasets["customer_churn"] = {
            "data": df,
            "metadata": metadata,
            "table_name": table_name,
            "table_metadata": table_metadata
        }
        
    print(f"✅ Generated {len(datasets)} datasets with Iceberg versioning")
    return datasets


def train_churn_model(iceberg_manager, lineage_tracker):
    """Train a churn prediction model from Iceberg table with lineage tracking."""
    table_name = "customer_churn"
    
    # Start lineage tracking for this job
    with lineage_tracker.track_job(
        namespace="ray_iceberg_demo",
        name="train_churn_model"
    ) as active_lineage:
        try:
            # Read from Iceberg table
            df = iceberg_manager.load_table_as_pandas(table_name, lineage_tracker=active_lineage)
            
            # Prepare features and target
            X = df.drop(columns=['CustomerId', 'Churned'])  # Changed from 'churn' to 'Churned'
            y = df['Churned']  # Changed from 'churn' to 'Churned'
            
            # Add one-hot encoding for categorical variables
            X = pd.get_dummies(X, columns=['ContractType'], drop_first=True)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Define model parameters 
            model_params = {
                'hidden_layer_sizes': (64, 32),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'batch_size': 64,
                'max_iter': 20,
                'early_stopping': True,
                'random_state': 42
            }
            
            # Add model parameters to lineage
            active_lineage.add_job_facet("model_params", model_params)
            
            # Train model
            model = MLPClassifier(**model_params)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            score = accuracy_score(y_test, y_pred)
            
            # Add model metrics to lineage
            active_lineage.add_job_facet("model_metrics", {
                "accuracy": float(score),
                "precision": float(precision_score(y_test, y_pred)),
                "recall": float(recall_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred))
            })
            
            # Save model
            os.makedirs("./storage/models", exist_ok=True)
            model_path = f"./storage/models/churn_model_{int(time.time())}.joblib"
            dump(model, model_path)
            
            print(f"✅ Model trained with accuracy: {score:.4f}")
            print(f"✅ Model saved to {model_path}")
            
            # Add model artifact to lineage
            active_lineage.add_output({
                "namespace": "ray_iceberg_demo", 
                "name": f"model://{os.path.basename(model_path)}",
                "facets": {
                    "model_info": {
                        "_producer": "ray_iceberg_demo",
                        "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/ModelFacet.json",
                        "name": "churn_mlp_classifier",
                        "version": "1.0.0"
                    }
                }
            })
            
            return model, model_path
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            raise


def train_models(args, env, datasets):
    """Train models using Ray Train."""
    print("\n=== Training Models with Ray Train ===")
    
    iceberg_manager = env["iceberg_manager"]
    model_registry = env["model_registry"]
    
    models = {}
    
    # Train customer churn model
    with LineageTracker(
        client=env["openlineage_client"],
        job_name="train_churn_model",
        job_description=f"Train customer churn prediction model v{args.model_version}"
    ) as lineage:
        # Get dataset information
        dataset_info = datasets["customer_churn"]
        table_name = dataset_info["table_name"]
        
        # Read dataset from Iceberg table
        df = iceberg_manager.load_table_as_pandas(
            table_name=table_name,
            lineage_tracker=lineage
        )
        
        # Prepare data
        target_column = "Churned"  # Changed from "churn" to "Churned"
        feature_columns = [col for col in df.columns if col not in [target_column, "CustomerId"]]
        
        X = df[feature_columns]
        y = df[target_column]
        
        # Create trainer
        trainer = SklearnModelTrainer(
            model_name="churn_predictor",
            model_type=args.model_type,
            model_version=args.model_version,
            random_state=42
        )
        
        # Track model config
        lineage.add_job_facet("model_config", {
            "model_type": args.model_type,
            "model_version": args.model_version,
            "feature_columns": feature_columns,
            "target_column": target_column
        })
        
        # Train model
        print(f"Training {args.model_type} model for churn prediction...")
        training_start = time.time()
        metrics = trainer.train(
            dataset=ray.data.from_pandas(df),
            target_column=target_column,
            feature_columns=feature_columns
        )
        training_duration = time.time() - training_start
        
        # Track metrics
        lineage.add_job_facet("training_metrics", {
            "duration_seconds": training_duration,
            **metrics
        })
        
        # Register model
        model_name = "churn_predictor"
        model_info = {
            "name": model_name,
            "version": args.model_version,
            "type": args.model_type,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "metrics": metrics,
            "training_duration": training_duration,
            "dataset_version": args.dataset_version,
            "created_at": time.time()
        }
        
        # Track model as output
        lineage.add_output({
            "namespace": args.namespace,
            "name": f"model://{model_name}",
            "facets": {
                "version": args.model_version,
                "metrics": metrics,
                "model_type": args.model_type,
                "feature_count": len(feature_columns)
            }
        })
        
        models["churn_predictor"] = {
            "model": trainer.model,
            "metrics": metrics,
            "metadata": model_info,
            "path": f"./models/churn_predictor_{args.model_version}.pkl",
            "feature_columns": feature_columns
        }
        
    print(f"✅ Trained and versioned {len(models)} models")
    return models


def deploy_models(args, env, models):
    """Deploy models to Ray Serve."""
    print("\n=== Deploying Models with Ray Serve ===")
    
    deployments = {}
    
    # Create Ray Serve deployer
    serve_host = f"http://localhost:{args.serve_port}"
    serve_deployer = RayServeDeployer(serve_host=serve_host)
    
    # Deploy churn prediction model
    with LineageTracker(
        client=env["openlineage_client"],
        job_name="deploy_churn_model",
        job_description=f"Deploy churn prediction model v{args.model_version} to Ray Serve"
    ) as lineage:
        # Get model info
        model_info = models["churn_predictor"]
        model_path = model_info["path"]
        model_metadata_path = model_path.replace(".pkl", "_metadata.json")
        
        # Track model as input
        lineage.add_input_dataset(
            namespace=args.namespace,
            name="model://churn_predictor",
            facets={
                "version": args.model_version,
                "metrics": model_info["metrics"],
                "path": model_path
            }
        )
        
        # Deploy model
        print("Deploying churn prediction model to Ray Serve...")
        deployment_info = serve_deployer.deploy_model(
            model_path=model_path,
            metadata_path=model_metadata_path,
            deployment_name="churn_predictor",
            route_prefix="/predict/churn",
            num_replicas=1,
            feature_columns=model_info["feature_columns"],
            return_probabilities=True
        )
        
        # Track deployment info
        lineage.add_job_facet("deployment_info", {
            "deployment_name": deployment_info["deployment_name"],
            "route_prefix": deployment_info["route_prefix"],
            "endpoint": deployment_info["endpoint"],
            "num_replicas": deployment_info["num_replicas"],
            "serve_host": serve_host
        })
        
        # Track deployment as output
        lineage.add_output_dataset(
            namespace=args.namespace,
            name=f"deployment://churn_predictor",
            facets={
                "version": args.model_version,
                "deployment_name": deployment_info["deployment_name"],
                "endpoint": deployment_info["endpoint"],
                "route_prefix": deployment_info["route_prefix"]
            }
        )
        
        deployments["churn_predictor"] = deployment_info
        
    print(f"✅ Deployed {len(deployments)} models to Ray Serve")
    return deployments


def test_model_endpoint(args, deployment_info):
    """Test the deployed model endpoint."""
    print("\n=== Testing Model Endpoint ===")
    
    import requests
    import random
    
    serve_host = f"http://localhost:{args.serve_port}"
    endpoint_url = deployment_info["endpoint"]
    
    print(f"Sending test request to: {endpoint_url}")
    
    # Create a sample customer record
    test_data = {
        "CustomerId": f"test-{random.randint(1000, 9999)}",
        "Age": random.randint(18, 80),
        "Tenure": random.randint(1, 20),
        "ContractType": random.choice(["Month-to-month", "One year", "Two year"]),
        "MonthlyCharges": random.randint(50, 200),
        "TotalCharges": random.randint(500, 5000),
        "HasPhoneService": random.choice([True, False]),
        "HasInternetService": random.choice([True, False])
    }
    
    try:
        response = requests.post(endpoint_url, json=test_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful!")
            
            # Safely access nested values with fallbacks for each level
            predictions = result.get('predictions', [None])
            probabilities = result.get('probabilities', [0.0])
            
            # Access model info with safe navigation
            metadata = result.get('metadata', {})
            model_info = metadata.get('model_info', {})
            model_name = model_info.get('model_name', 'Unknown')
            
            print(f"Prediction: {predictions[0]}")
            if probabilities[0] is not None:
                print(f"Probability: {probabilities[0]:.4f}")
            else:
                print(f"Probability: N/A")
            print(f"Model: {model_name}")
            return True
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Error testing endpoint: {e}")
        return False


def check_marquez_connectivity(url):
    """Check if Marquez API is reachable."""
    import requests
    
    # Extract the base URL without the lineage endpoint
    base_url = url
    if "/api/v1/lineage" in url:
        base_url = url.replace("/api/v1/lineage", "")
    
    try:
        # First try the health endpoint
        health_response = requests.get(f"{base_url}/api/v1/health", timeout=5)
        if health_response.status_code == 200:
            return True
        
        # If health endpoint returns 404, try the namespaces endpoint
        # This is because the health endpoint may not exist in all versions
        namespaces_response = requests.get(f"{base_url}/api/v1/namespaces", timeout=5)
        if namespaces_response.status_code == 200:
            return True
        
        return False
    except Exception as e:
        print(f"Error connecting to Marquez: {e}")
        return False


def cleanup_ray(args):
    """Cleanup Ray resources."""
    print("\n=== Cleaning up Ray Resources ===")
    
    if ray.is_initialized():
        print("Shutting down Ray...")
        ray.shutdown()
    
    print("Ray shutdown complete. To completely stop Ray processes, run:")
    print("ray stop")


def main():
    """Run the demo."""
    args = parse_args()
    
    print("=" * 60)
    print("Simplified Ray + Iceberg + OpenLineage Demo")
    print("=" * 60)
    
    # Check Marquez connectivity
    print("\n=== Checking Marquez Connectivity ===")
    if not check_marquez_connectivity(args.openlineage_url):
        print(f"❌ Cannot connect to Marquez at {args.openlineage_url}")
        print("Make sure Marquez is running with Docker Compose:")
        print("docker-compose up -d postgres marquez web")
        print("If Marquez is running, check the URL and try again.")
        return
    else:
        print(f"✅ Connected to Marquez at {args.openlineage_url}")
    
    try:
        # Step 1: Set up environment
        try:
            env = setup_environment(args)
            if not env:
                print("\n❌ Environment setup failed")
                return
        except Exception as e:
            print(f"\n❌ Environment setup failed: {e}")
            cleanup_ray(args)
            return
        
        # Step 2: Generate datasets
        datasets = generate_datasets(args, env)
        
        # Step 3: Train models
        models = train_models(args, env, datasets)
        
        # Step 4: Deploy models
        deployments = deploy_models(args, env, models)
        
        # Step 5: Test model endpoint
        if deployments:
            test_model_endpoint(args, deployments["churn_predictor"])
        
        # Display summary
        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print("=" * 60)
        
        # Display access points
        print("\nAccess Points:")
        try:
            dashboard_url = f"http://127.0.0.1:{args.dashboard_port}" if ray.is_initialized() else f"http://localhost:{args.dashboard_port}"
            print(f"Ray Dashboard: {dashboard_url}")
        except:
            print("Ray Dashboard: URL not available")
        print(f"Ray Serve Endpoint: http://localhost:{args.serve_port}/predict/churn")
        print(f"Marquez UI: {args.marquez_web_url}")
        print(f"View your data lineage at: {args.marquez_web_url}/namespaces/{args.namespace}")
        
        # Keep Ray running for interaction with the endpoint
        print("\nRay cluster is still running so you can interact with the endpoint.")
        print("When you're done, press Ctrl+C to shutdown or run 'ray stop'.")
        
        # Wait until Ctrl+C
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            cleanup_ray(args)
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
        cleanup_ray(args)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        cleanup_ray(args)


if __name__ == "__main__":
    main() 