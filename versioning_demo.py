#!/usr/bin/env python3
"""
Versioning Demonstration for Ray + Iceberg + OpenLineage

This script demonstrates:
1. Creating a new version of the dataset with different parameters
2. "Time traveling" between dataset versions
3. Training a new model on the new dataset version
4. Comparing models trained on different dataset versions
"""

import os
import ray
import pandas as pd
import numpy as np
import time
import json
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Import components from src modules
from src.data.generators import DataGenerator
from src.data.iceberg_manager import IcebergManager
from src.models.trainer import SklearnModelTrainer
from src.lineage.openlineage_client import OpenLineageClient, LineageTracker

# Initialize Ray if not already initialized
if not ray.is_initialized():
    ray.init()

# Initialize the IcebergManager
iceberg_manager = IcebergManager(
    warehouse_path="./storage/iceberg_warehouse",
    namespace="ray_iceberg_demo",
    catalog_name="ray_demo_catalog"
)

# Initialize OpenLineage client
openlineage_client = OpenLineageClient(
    api_url="http://localhost:5002/api/v1/lineage",
    namespace="ray_iceberg_demo"
)

print("=" * 60)
print("Versioning Demonstration")
print("=" * 60)

# Part 1: List existing dataset versions
print("\n=== Existing Dataset Versions ===")
table_name = "customer_churn"
table_path = iceberg_manager.get_table_path(table_name)

# Find all snapshot directories
snapshots = [d for d in os.listdir(table_path) if d.startswith("snapshot-")]
snapshots.sort()  # Sort by timestamp

print(f"Found {len(snapshots)} versions of {table_name} dataset:")
for snapshot in snapshots:
    snapshot_id = snapshot.replace("snapshot-", "")
    metadata_file = f"./storage/iceberg_warehouse/_metadata/{table_name}-{snapshot_id}.json"
    
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            print(f"  - Version {snapshot_id}: {metadata.get('rows', 'unknown')} rows, created at {time.ctime(int(snapshot_id))}")
    else:
        print(f"  - Version {snapshot_id}: metadata not found")

# Get current version from LATEST file
with open(os.path.join(table_path, "LATEST"), 'r') as f:
    current_version = f.read().strip()
    print(f"\nCurrent version is: {current_version}")

# Part 2: Create a new dataset version with different parameters
print("\n=== Creating New Dataset Version ===")
# Track lineage for the dataset creation
with LineageTracker(
    client=openlineage_client,
    job_name="generate_customer_dataset",
    job_description="Generate customer churn dataset v2.0.0 with higher noise"
) as lineage:
    # Generate dataset with different parameters
    print("Generating customer churn dataset v2.0.0 with higher noise...")
    df, metadata = DataGenerator.generate_customer_dataset(
        n_rows=12000,  # More rows
        version="2.0.0",  # New version
        random_seed=43,  # Different seed
        noise_level=0.8,  # Higher noise
        as_ray_dataset=False
    )
    
    # Update metadata
    metadata["version"] = "2.0.0"
    metadata["description"] = "Synthetic customer churn dataset v2.0.0 with higher noise"
    
    # Track dataset as input to this job
    lineage.add_job_facet("generator_config", {
        "n_rows": 12000,
        "version": "2.0.0",
        "random_seed": 43,
        "noise_level": 0.8,
        "job_run_description": "Second version of customer dataset with higher noise"
    })
    
    # Store dataset in Iceberg
    table_metadata = iceberg_manager.create_table(
        table_name=table_name,
        df=df,
        metadata=metadata,
        lineage_tracker=lineage
    )
    
    # Show new version info
    new_version = table_metadata["snapshot_id"]
    print(f"Created new version: {new_version}")

# Part 3: Time travel between versions
print("\n=== Time Travel Between Dataset Versions ===")

# Load the original version
print(f"Loading original version ({current_version})...")
df_original = iceberg_manager.load_table_as_pandas(
    table_name=table_name,
    snapshot_id=int(current_version)
)
print(f"Loaded {len(df_original)} rows from original version")
print(f"Sample from original version (first 3 rows):")
print(df_original.head(3))
print(f"Churn rate in original version: {df_original['Churned'].mean():.2%}")

# Load the new version
print(f"\nLoading new version ({new_version})...")
df_new = iceberg_manager.load_table_as_pandas(
    table_name=table_name,
    snapshot_id=int(new_version)
)
print(f"Loaded {len(df_new)} rows from new version")
print(f"Sample from new version (first 3 rows):")
print(df_new.head(3))
print(f"Churn rate in new version: {df_new['Churned'].mean():.2%}")

# Part 4: Train a new model on the new dataset version
print("\n=== Training New Model on New Dataset Version ===")

# Track lineage for model training
with LineageTracker(
    client=openlineage_client,
    job_name="train_churn_model",
    job_description="Train customer churn prediction model v2.0.0 on dataset v2.0.0"
) as lineage:
    # Create Ray Dataset from pandas DataFrame
    new_dataset = ray.data.from_pandas(df_new)
    
    # Prepare data
    target_column = "Churned"
    feature_columns = [col for col in df_new.columns if col not in [target_column, "CustomerId"]]
    
    # Create trainer with different hyperparameters
    trainer = SklearnModelTrainer(
        model_name="churn_predictor",
        model_type="random_forest",
        model_version="2.0.0",
        random_state=43,
        hyperparameters={
            "n_estimators": 150,  # More trees
            "max_depth": 15,      # Deeper trees
        }
    )
    
    # Track model config
    lineage.add_job_facet("model_config", {
        "model_type": "random_forest",
        "model_version": "2.0.0",
        "feature_columns": feature_columns,
        "target_column": target_column,
        "hyperparameters": {
            "n_estimators": 150,
            "max_depth": 15,
        },
        "job_run_description": "Second version of model with more complex hyperparameters"
    })
    
    # Train model
    print(f"Training random_forest model v2.0.0 for churn prediction...")
    metrics = trainer.train(
        dataset=new_dataset,
        target_column=target_column,
        feature_columns=feature_columns
    )
    
    # Show metrics
    print(f"Model v2.0.0 metrics:")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall: {metrics['recall']:.4f}")
    print(f"  - F1 Score: {metrics['f1']:.4f}")
    print(f"  - ROC AUC: {metrics.get('roc_auc', 'N/A')}")

# Part 5: Compare models
print("\n=== Comparing Models Trained on Different Dataset Versions ===")

# Load model metadata
with open('./models/churn_predictor_1.0.0_metadata.json', 'r') as f:
    model_v1_metadata = json.load(f)

with open('./models/churn_predictor_2.0.0_metadata.json', 'r') as f:
    model_v2_metadata = json.load(f)

# Print comparison
print("Model Comparison:")
print("                  | Model v1.0.0 | Model v2.0.0")
print("--------------------------------------------------")
print(f"Dataset Version  | 1.0.0        | 2.0.0")
print(f"Dataset Size     | {model_v1_metadata['model_info']['training_samples']} rows    | {model_v2_metadata['model_info']['training_samples']} rows")
print(f"Accuracy         | {model_v1_metadata['accuracy']:.4f}        | {model_v2_metadata['accuracy']:.4f}")
print(f"Precision        | {model_v1_metadata['precision']:.4f}        | {model_v2_metadata['precision']:.4f}")
print(f"Recall           | {model_v1_metadata['recall']:.4f}        | {model_v2_metadata['recall']:.4f}")
print(f"F1 Score         | {model_v1_metadata['f1']:.4f}        | {model_v2_metadata['f1']:.4f}")
print(f"ROC AUC          | {model_v1_metadata.get('roc_auc', 'N/A')}        | {model_v2_metadata.get('roc_auc', 'N/A')}")

print("\n=== Summary ===")
print("This demonstration showed:")
print("1. Listing existing dataset versions")
print("2. Creating a new version of the dataset with different parameters")
print("3. Time traveling between dataset versions")
print("4. Training a new model on the new dataset version")
print("5. Comparing models trained on different dataset versions")
print("\nAll of this was tracked with OpenLineage, capturing the full lineage of data and models.")
print("You can view the lineage graph at http://localhost:3000/namespaces/ray_iceberg_demo")

# Shutdown Ray
ray.shutdown() 