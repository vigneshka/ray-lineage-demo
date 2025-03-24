# Ray + OpenLineage Demo

This demo showcases a complete end-to-end MLOps pipeline with data versioning, model training, serving, and lineage tracking:

1. **Ray Data** used to generate datasets with Iceberg versioning
2. **Ray Train** to generate versioned ML models 
3. **Ray Serve** serving multiple model versions for predictions
4. **OpenLineage** tracking with Marquez for complete data and model lineage visualization

## What is OpenLineage?

[OpenLineage](https://openlineage.io/) is an open framework for data lineage collection and analysis. It:

- Tracks metadata and provenance for datasets and jobs
- Provides standardized definitions for data lineage
- Offers integrations with many data platforms and tools
- Enables end-to-end data lineage across your entire stack

**Marquez** is the reference implementation of OpenLineage, providing:
- A server for collecting lineage events
- Storage for lineage metadata
- APIs for querying lineage
- A web UI for visualizing lineage graphs
- Developed originally by WeWork and now maintained by the Linux Foundation

## Architecture

```
┌─────────────────────────────────┐
│ Local Machine                   │
│                                 │
│ ┌─────────────┐  ┌────────────┐ │
│ │ Ray Cluster │  │ ML Models  │ │
│ │             │◄─┤            │ │
│ │ - Ray Data  │  │ - Training │ │
│ │ - Ray Train │──► - Serving  │ │
│ │ - Ray Serve │  │            │ │
│ └─────┬───▲───┘  └────────────┘ │
│       │   │                     │
│       │   │     ┌─────────────┐ │
│       │   │     │ Iceberg     │ │
│       │   │     │ Catalog     │ │
│       │   └─────┤ - Datasets  │ │
│       │         └─────────────┘ │
└───────┼─────────────────────────┘
        │
        │ Lineage Events
        ▼
┌───────────────────────────────┐
│ Docker                        │
│                               │
│ ┌───────────┐  ┌───────────┐  │
│ │ Marquez   │  │ Postgres  │  │
│ │           │◄─┤           │  │
│ │ OpenLineage│  │  Database│  │
│ └───────────┘  └───────────┘  │
│                               │
└───────────────────────────────┘
```

## Technology Stack

### Ray
[Ray](https://ray.io/) is an open-source unified compute framework that makes it easy to scale AI and Python workloads. In this demo we use:

- **Ray Data**: For distributed data processing
- **Ray Train**: For distributed model training
- **Ray Serve**: For model serving and deployment

### Apache Iceberg
[Apache Iceberg](https://iceberg.apache.org/) is an open table format designed for large analytical datasets. Key features:

- Schema evolution
- Partition evolution
- Time travel queries
- Version rollback
- ACID transactions

In this demo, we use Ray's native integration with Iceberg to manage dataset versions.

### Requirements

- Python 3.8+
- Docker and Docker Compose
- Required Python packages (installed via requirements.txt):
  - ray[data,train,serve]==2.44.0
  - openlineage-python==1.8.0
  - tensorflow
  - pandas, numpy, etc.

## Setup Instructions

### Automated Setup (Recommended)

The easiest way to set up the environment is to use the provided setup script:

```bash
# Make the script executable if needed
chmod +x setup_env.sh

# Run the setup script
./setup_env.sh
```

This script will:
1. Create and activate a Python virtual environment
2. Install all required dependencies
3. Verify TensorFlow installation
4. Check Docker and Docker Compose availability
5. Start Marquez if requested

### Manual Setup

If you prefer to set up manually, follow these steps:

#### 1. Clone the Repository

```bash
git clone git@github.com:vigneshka/ray-lineage-demo.git
cd ray-lineage-demo
```

#### 2. Set Up Python Virtual Environment

It's recommended to use a virtual environment:

```bash
# Create a virtual environment
python3 -m venv ray_lineage_venv

# Activate the virtual environment
# On macOS/Linux:
source ray_lineage_venv/bin/activate
# On Windows:
# ray_lineage_venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

#### 3. Start Marquez with Docker Compose

```bash
# Start Marquez and related components
docker-compose up -d
```

This starts:
- PostgreSQL database
- Marquez API server
- Marquez Web UI

Marquez will be available at:
- Web UI: http://localhost:3000
- API: http://localhost:5002

## Verify Setup

After setting up, you can verify that everything is working correctly:

```bash
# Make the verification script executable
chmod +x verify_setup.py

# Run the verification script
./verify_setup.py
```

The verification script checks:
- Python virtual environment
- Required dependencies
- Ray cluster functionality
- Marquez API connectivity
- Directory structure

If all checks pass, your environment is ready to run the demo.

## Running the Demo

The demo consists of a sequence of scripts that build on each other to demonstrate the complete MLOps lifecycle:

```bash
# Step 1: Run the main pipeline (creates initial dataset and first model version)
python ray_iceberg_lineage.py

# Step 2: Create a new dataset version and train a new model
python versioning_demo.py

# Step 3: Serve both model versions and compare predictions
python serve_models_demo.py
```

Each script demonstrates different aspects of the MLOps pipeline:

1. **ray_iceberg_lineage.py** builds the foundation:
   - Initializes the Iceberg catalog
   - Creates the first dataset version
   - Trains the first model version (v1.0.0)
   - Tracks lineage with OpenLineage

2. **versioning_demo.py** demonstrates versioning:
   - Creates a second dataset version with different characteristics
   - Trains a second model version (v2.0.0) with different hyperparameters
   - Shows how to query dataset versions and compare models

3. **serve_models_demo.py** demonstrates model serving:
   - Deploys both model versions to Ray Serve
   - Allows for side-by-side prediction comparison
   - Enables A/B testing between models

You can run the full sequence to see the complete workflow, or run individual scripts if you're interested in specific aspects.

## What You'll See

The demo demonstrates a complete MLOps pipeline:

1. **Data Management**:
   - Customer churn data generation
   - Dataset versioning with Iceberg
   - Time travel queries between dataset versions

2. **Model Training**:
   - Training on different dataset versions
   - Hyperparameter tuning
   - Model versioning and metadata tracking

3. **Model Serving**:
   - Deploying multiple model versions concurrently
   - A/B testing between versions
   - Comparing model predictions

4. **Lineage Tracking**:
   - Dataset-to-model relationships
   - Data transformation tracking
   - Model metadata and performance metrics

You can view:
- Ray Dashboard: http://localhost:8265
- Marquez UI: http://localhost:3000
- Ray Serve API: http://localhost:8000/predict/v1 (model v1.0.0)
- Ray Serve API: http://localhost:8000/predict/v2 (model v2.0.0)

## Demo Purpose and Expected Observations

### Purpose of the Demo

This demo was specifically designed to showcase:

1. **End-to-End Lineage in ML Workflows**: How data flows from generation to model deployment, with every transformation tracked
2. **Version Management**: How to properly version both datasets and models while maintaining their relationship
3. **Reproducibility**: The ability to reconstruct the exact path from data to deployed model
4. **Production ML Practices**: A simplified but realistic implementation of ML in production with versioning, A/B testing, and lineage

### What to Observe in Marquez

When you access the Marquez UI at http://localhost:3000, you should observe:

1. **Job Nodes**:
   - `generate_customer_dataset`: Shows different runs for different versions of the dataset
   - `train_churn_model`: Shows model training runs for different versions
   - `deploy_churn_model`: Shows model deployment information
   
2. **Dataset Nodes**:
   - `customer_churn`: The versioned datasets in Iceberg
   - `model://churn_predictor`: The trained model artifacts
   - `deployment://churn_predictor`: The deployed model endpoints

3. **Lineage Graphs**:
   - The complete flow from data generation → training → serving
   - Version relationships between datasets and models
   - Clear distinction between different runs of the same job

4. **Run Metadata**:
   - Dataset metrics (row counts, schema info)
   - Model hyperparameters
   - Performance metrics for each model version
   - Timestamps showing when each operation occurred

Pay special attention to how consistent job naming (rather than creating new job names for each version) creates cleaner lineage graphs that better show the evolution of your ML assets over time.

## Benefits of OpenLineage in ML Pipelines

OpenLineage provides several key benefits in ML workflows:

1. **Data Governance**: Track where data comes from, how it's transformed, and where it goes
2. **Reproducibility**: Know exactly which dataset version was used to train each model version
3. **Debugging**: When issues arise, trace back through the lineage to find root causes
4. **Compliance**: Maintain detailed records for regulatory requirements
5. **Impact Analysis**: Understand how changes to datasets affect downstream models

## Troubleshooting

### TensorFlow Dependency

If you encounter issues related to TensorFlow, you may need to install it separately:

```bash
pip install tensorflow>=2.10.0
```

### Ray Cluster Issues

If Ray has trouble starting:

```bash
# Stop any existing Ray processes
ray stop
# Then restart the demo
python ray_iceberg_lineage.py
```

### Marquez Connection Issues

If the demo can't connect to Marquez:

```bash
# Check if Marquez is running
curl http://localhost:5002/api/v1/namespaces
# If not, restart the containers
docker-compose restart
```

## Cleanup

After running the demos, you'll want to clean up the resources that were created:

### 1. Stop Running Processes

```bash
# Stop any running Ray processes
ray stop

# Stop Marquez and clean up Docker containers
docker-compose down
```

### 2. Remove Generated Files

```bash
# Remove the Iceberg warehouse directory (contains dataset files)
rm -rf ./storage/iceberg_warehouse

# Remove trained models
rm -rf ./models

# Remove Python cache files
rm -rf ./src/__pycache__
rm -rf ./src/*/__pycache__

# Remove logs
rm -f output.log
rm -f *.log

# Remove any temporary or cache files
rm -rf ./.ray
```

### 3. Virtual Environment

If you want to completely remove the virtual environment:

```bash
# First, deactivate it if it's active
deactivate

# Then remove the directory
rm -rf ray_lineage_venv
```

This cleanup process ensures that you can start fresh the next time you run the demo. 
