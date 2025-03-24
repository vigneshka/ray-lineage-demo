"""
OpenLineage Client for Ray + Iceberg Integration
This module provides OpenLineage tracking capabilities for the unified demo.
"""

import os
import json
import uuid
import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import requests
import pandas as pd
import ray


class OpenLineageClient:
    """Client for interacting with the OpenLineage API."""
    
    def __init__(self, api_url: str = "http://localhost:5002/api/v1/lineage", namespace: str = "ray_iceberg_demo"):
        """Initialize the OpenLineage client.
        
        Args:
            api_url: URL for the OpenLineage API
            namespace: Default namespace for jobs and datasets
        """
        self.api_url = api_url
        self.namespace = namespace
        self.producer = "ray_iceberg_lineage_demo"
    
    def create_lineage_event(self, 
                             job_name: str, 
                             event_type: str = "START", 
                             job_namespace: Optional[str] = None,
                             inputs: Optional[List[Dict]] = None, 
                             outputs: Optional[List[Dict]] = None,
                             job_facets: Optional[Dict] = None,
                             parent_run_id: Optional[str] = None,
                             run_id: Optional[str] = None) -> Tuple[Dict, str]:
        """Create an OpenLineage event.
        
        Args:
            job_name: Name of the job
            event_type: Type of event (START, COMPLETE, FAIL, etc.)
            job_namespace: Namespace for the job
            inputs: List of input datasets
            outputs: List of output datasets
            job_facets: Dictionary of job facets
            parent_run_id: Parent run ID for linking events
            run_id: Run ID for the event
            
        Returns:
            Tuple of (event, run_id)
        """
        # Fallback to defaults
        job_namespace = job_namespace or self.namespace
        inputs = inputs or []
        outputs = outputs or []
        job_facets = job_facets or {}
        
        # Generate or use provided run ID
        if event_type == "START" or not run_id:
            run_id = run_id or str(uuid.uuid4())
        
        # Create the event
        event = {
            "eventType": event_type,
            "eventTime": datetime.datetime.utcnow().isoformat() + "Z",
            "run": {
                "runId": run_id
            },
            "job": {
                "namespace": job_namespace,
                "name": job_name,
                "facets": job_facets
            },
            "inputs": inputs,
            "outputs": outputs,
            "producer": self.producer
        }
        
        return event, run_id
    
    def send_lineage_event(self, event: Dict) -> bool:
        """Send an OpenLineage event to the API.
        
        Args:
            event: OpenLineage event to send
            
        Returns:
            True if the event was sent successfully, False otherwise
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=event)
            if response.status_code == 201:
                print(f"Successfully sent {event['eventType']} event for {event['job']['namespace']}.{event['job']['name']}")
                print(f"  Run ID: {event['run']['runId']}")
                print(f"  Inputs: {len(event['inputs'])}, Outputs: {len(event['outputs'])}")
                return True
            else:
                print(f"Failed to send event: Status code {response.status_code}")
                print(f"Response: {response.text}")
                return False
        except Exception as e:
            print(f"Error sending lineage event: {e}")
            return False
    
    def track_dataset_version(self, 
                             dataset_name: str, 
                             dataset_version: str,
                             dataset_path: str,
                             schema: Optional[List[Dict]] = None,
                             description: Optional[str] = None,
                             custom_facets: Optional[Dict] = None) -> Dict:
        """Create a dataset schema for OpenLineage.
        
        Args:
            dataset_name: Name of the dataset
            dataset_version: Version of the dataset
            dataset_path: Path to the dataset
            schema: List of field definitions
            description: Dataset description
            custom_facets: Custom facets to add
            
        Returns:
            OpenLineage dataset schema
        """
        dataset = {
            "namespace": self.namespace,
            "name": dataset_name
        }
        
        facets = {}
        
        # Add documentation facet if description is provided
        if description:
            facets["documentation"] = {
                "_producer": self.producer,
                "_schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json#/definitions/DocumentationDatasetFacet",
                "description": description
            }
        
        # Add schema facet if schema is provided
        if schema:
            facets["schema"] = {
                "_producer": self.producer,
                "_schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json#/definitions/SchemaDatasetFacet",
                "fields": schema
            }
        
        # Add datasource facet
        facets["dataSource"] = {
            "_producer": self.producer,
            "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DatasourceDatasetFacet.json",
            "name": os.path.basename(dataset_path),
            "uri": f"file://{os.path.abspath(dataset_path)}"
        }
        
        # Add version facet
        facets["version"] = {
            "_producer": self.producer,
            "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/VersionDatasetFacet.json",
            "version": dataset_version
        }
        
        # Add dataset properties from metadata format (Iceberg-like metadata)
        facets["datasetProperties"] = {
            "_producer": self.producer,
            "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DatasourceDatasetFacet.json",
            "properties": {
                "format": "parquet",  # Can be adjusted based on actual format
                "snapshot_id": str(uuid.uuid4()),
                "schema_id": 0,
                "partition_spec_id": 0
            }
        }
        
        # Add custom facets
        if custom_facets:
            facets.update(custom_facets)
        
        dataset["facets"] = facets
        return dataset
    
    def track_ray_dataset(self, 
                         ds: ray.data.Dataset,
                         dataset_name: str,
                         dataset_version: str,
                         description: Optional[str] = None,
                         custom_facets: Optional[Dict] = None) -> Dict:
        """Track a Ray Dataset as an OpenLineage dataset.
        
        Args:
            ds: Ray Dataset
            dataset_name: Name for the dataset
            dataset_version: Version of the dataset
            description: Dataset description
            custom_facets: Additional facets
            
        Returns:
            OpenLineage dataset schema
        """
        # Get schema from Ray Dataset
        df_sample = ds.take(1)
        if df_sample:
            schema = [{
                "name": col,
                "type": str(type(df_sample[0][col]).__name__)
            } for col in df_sample[0].keys()]
        else:
            schema = []
        
        # Add Ray-specific facets
        ray_facets = {
            "rayDataset": {
                "_producer": self.producer,
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/CustomFacet.json",
                "type": "ray.data.Dataset",
                "nrows": ds.count(),
                "nblocks": ds.num_blocks(),
                "size_bytes": ds.size_bytes()
            }
        }
        
        # Combine with custom facets
        all_facets = custom_facets or {}
        all_facets.update(ray_facets)
        
        # Use a virtual path since Ray datasets might not have a physical path
        virtual_path = f"ray://{self.namespace}/{dataset_name}/{dataset_version}"
        
        return self.track_dataset_version(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            dataset_path=virtual_path,
            schema=schema,
            description=description,
            custom_facets=all_facets
        )
    
    def track_model_version(self,
                          model_name: str,
                          model_version: str,
                          model_path: str,
                          model_type: str,
                          metrics: Optional[Dict[str, float]] = None,
                          hyperparams: Optional[Dict[str, Any]] = None,
                          description: Optional[str] = None,
                          custom_facets: Optional[Dict] = None) -> Dict:
        """Track a model as an OpenLineage dataset.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            model_path: Path to the model artifact
            model_type: Type of the model (e.g., "RandomForest", "NeuralNetwork")
            metrics: Model performance metrics
            hyperparams: Model hyperparameters
            description: Model description
            custom_facets: Additional facets
            
        Returns:
            OpenLineage dataset schema
        """
        # Start with basic dataset information
        model_dataset = self.track_dataset_version(
            dataset_name=model_name,
            dataset_version=model_version,
            dataset_path=model_path,
            description=description,
            custom_facets=custom_facets
        )
        
        # Add model-specific facets
        model_facets = {
            "model": {
                "_producer": self.producer,
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/CustomFacet.json",
                "type": model_type,
                "metrics": metrics or {},
                "hyperparams": hyperparams or {}
            }
        }
        
        # Update the model dataset facets
        if "facets" not in model_dataset:
            model_dataset["facets"] = {}
        
        model_dataset["facets"].update(model_facets)
        
        return model_dataset
    
    def track_data_generation(self,
                             job_name: str,
                             output_dataset: Dict,
                             job_description: Optional[str] = None,
                             run_id: Optional[str] = None) -> str:
        """Track a data generation job.
        
        Args:
            job_name: Name of the job
            output_dataset: Output dataset schema
            job_description: Description of the job
            run_id: Optional run ID
            
        Returns:
            Run ID for the job
        """
        job_facets = {}
        
        # Add documentation facet
        if job_description:
            job_facets["documentation"] = {
                "_producer": self.producer,
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DocumentationJobFacet.json",
                "description": job_description
            }
        
        # START event
        start_event, run_id = self.create_lineage_event(
            job_name=job_name,
            event_type="START",
            outputs=[output_dataset],
            job_facets=job_facets,
            run_id=run_id
        )
        
        # Send START event
        self.send_lineage_event(start_event)
        
        return run_id
    
    def complete_data_generation(self,
                               job_name: str,
                               output_dataset: Dict,
                               run_id: str,
                               additional_facets: Optional[Dict] = None) -> None:
        """Complete data generation tracking.
        
        Args:
            job_name: Name of the job
            output_dataset: Output dataset schema
            run_id: Run ID from start event
            additional_facets: Additional job facets
        """
        job_facets = additional_facets or {}
        
        # COMPLETE event
        complete_event, _ = self.create_lineage_event(
            job_name=job_name,
            event_type="COMPLETE",
            outputs=[output_dataset],
            job_facets=job_facets,
            parent_run_id=run_id
        )
        
        # Send COMPLETE event
        self.send_lineage_event(complete_event)
    
    def track_data_transformation(self,
                                job_name: str,
                                input_datasets: List[Dict],
                                output_datasets: List[Dict],
                                job_description: Optional[str] = None,
                                job_facets: Optional[Dict] = None,
                                run_id: Optional[str] = None) -> str:
        """Track a data transformation job.
        
        Args:
            job_name: Name of the job
            input_datasets: Input datasets
            output_datasets: Output datasets
            job_description: Description of the job
            job_facets: Additional job facets
            run_id: Optional run ID
            
        Returns:
            Run ID for the transformation
        """
        all_facets = job_facets or {}
        
        # Add documentation facet
        if job_description and "documentation" not in all_facets:
            all_facets["documentation"] = {
                "_producer": self.producer,
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DocumentationJobFacet.json",
                "description": job_description
            }
        
        # START event
        start_event, run_id = self.create_lineage_event(
            job_name=job_name,
            event_type="START",
            inputs=input_datasets,
            outputs=output_datasets,
            job_facets=all_facets,
            run_id=run_id
        )
        
        # Send START event
        self.send_lineage_event(start_event)
        
        return run_id
    
    def complete_data_transformation(self,
                                   job_name: str,
                                   input_datasets: List[Dict],
                                   output_datasets: List[Dict],
                                   run_id: str,
                                   additional_facets: Optional[Dict] = None) -> None:
        """Complete a data transformation job.
        
        Args:
            job_name: Name of the job
            input_datasets: Input datasets
            output_datasets: Output datasets
            run_id: Run ID from start event
            additional_facets: Additional job facets
        """
        job_facets = additional_facets or {}
        
        # COMPLETE event
        complete_event, _ = self.create_lineage_event(
            job_name=job_name,
            event_type="COMPLETE",
            inputs=input_datasets,
            outputs=output_datasets,
            job_facets=job_facets,
            parent_run_id=run_id
        )
        
        # Send COMPLETE event
        self.send_lineage_event(complete_event)
    
    def track_model_training(self,
                           job_name: str,
                           input_datasets: List[Dict],
                           output_model: Dict,
                           hyperparams: Dict,
                           job_description: Optional[str] = None,
                           run_id: Optional[str] = None) -> str:
        """Track a model training job.
        
        Args:
            job_name: Name of the job
            input_datasets: Input datasets
            output_model: Output model
            hyperparams: Hyperparameters
            job_description: Description of the job
            run_id: Optional run ID
            
        Returns:
            Run ID for the training
        """
        job_facets = {
            "hyperparameters": {
                "_producer": self.producer,
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/HyperparametersJobFacet.json",
                "hyperparameters": hyperparams
            }
        }
        
        # Add documentation facet
        if job_description:
            job_facets["documentation"] = {
                "_producer": self.producer,
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DocumentationJobFacet.json",
                "description": job_description
            }
        
        # START event
        start_event, run_id = self.create_lineage_event(
            job_name=job_name,
            event_type="START",
            inputs=input_datasets,
            outputs=[output_model],
            job_facets=job_facets,
            run_id=run_id
        )
        
        # Send START event
        self.send_lineage_event(start_event)
        
        return run_id
    
    def complete_model_training(self,
                              job_name: str,
                              input_datasets: List[Dict],
                              output_model: Dict,
                              run_id: str,
                              metrics: Dict,
                              additional_facets: Optional[Dict] = None) -> None:
        """Complete a model training job.
        
        Args:
            job_name: Name of the job
            input_datasets: Input datasets
            output_model: Output model
            run_id: Run ID from start event
            metrics: Performance metrics
            additional_facets: Additional job facets
        """
        job_facets = additional_facets or {}
        
        # Add metrics facet
        job_facets["metrics"] = {
            "_producer": self.producer,
            "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/MetricsJobFacet.json",
            "metrics": metrics
        }
        
        # COMPLETE event
        complete_event, _ = self.create_lineage_event(
            job_name=job_name,
            event_type="COMPLETE",
            inputs=input_datasets,
            outputs=[output_model],
            job_facets=job_facets,
            parent_run_id=run_id
        )
        
        # Send COMPLETE event
        self.send_lineage_event(complete_event)
    
    def track_model_deployment(self,
                             job_name: str,
                             input_model: Dict,
                             output_endpoint: Dict,
                             deployment_config: Dict,
                             job_description: Optional[str] = None,
                             run_id: Optional[str] = None) -> str:
        """Track a model deployment job.
        
        Args:
            job_name: Name of the job
            input_model: Input model
            output_endpoint: Output deployment endpoint
            deployment_config: Deployment configuration
            job_description: Description of the job
            run_id: Optional run ID
            
        Returns:
            Run ID for the deployment
        """
        job_facets = {
            "deployment": {
                "_producer": self.producer,
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/CustomFacet.json",
                "config": deployment_config
            }
        }
        
        # Add documentation facet
        if job_description:
            job_facets["documentation"] = {
                "_producer": self.producer,
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DocumentationJobFacet.json",
                "description": job_description
            }
        
        # START event
        start_event, run_id = self.create_lineage_event(
            job_name=job_name,
            event_type="START",
            inputs=[input_model],
            outputs=[output_endpoint],
            job_facets=job_facets,
            run_id=run_id
        )
        
        # Send START event
        self.send_lineage_event(start_event)
        
        return run_id
    
    def complete_model_deployment(self,
                                job_name: str,
                                input_model: Dict,
                                output_endpoint: Dict,
                                run_id: str,
                                deployment_status: Dict,
                                additional_facets: Optional[Dict] = None) -> None:
        """Complete a model deployment job.
        
        Args:
            job_name: Name of the job
            input_model: Input model
            output_endpoint: Output deployment endpoint
            run_id: Run ID from start event
            deployment_status: Deployment status information
            additional_facets: Additional job facets
        """
        job_facets = additional_facets or {}
        
        # Add deployment status facet
        job_facets["deploymentStatus"] = {
            "_producer": self.producer,
            "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/CustomFacet.json",
            "status": deployment_status
        }
        
        # COMPLETE event
        complete_event, _ = self.create_lineage_event(
            job_name=job_name,
            event_type="COMPLETE",
            inputs=[input_model],
            outputs=[output_endpoint],
            job_facets=job_facets,
            parent_run_id=run_id
        )
        
        # Send COMPLETE event
        self.send_lineage_event(complete_event)
    
    def track_inference(self,
                      job_name: str,
                      input_data: Dict,
                      model: Dict,
                      inference_details: Dict,
                      job_description: Optional[str] = None,
                      run_id: Optional[str] = None) -> str:
        """Track a model inference job.
        
        Args:
            job_name: Name of the job
            input_data: Input data
            model: Model used for inference
            inference_details: Details about the inference
            job_description: Description of the job
            run_id: Optional run ID
            
        Returns:
            Run ID for the inference
        """
        job_facets = {
            "inference": {
                "_producer": self.producer,
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/CustomFacet.json",
                "details": inference_details
            }
        }
        
        # Add documentation facet
        if job_description:
            job_facets["documentation"] = {
                "_producer": self.producer,
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DocumentationJobFacet.json",
                "description": job_description
            }
        
        # Output is virtual, representing the inference result
        output_result = {
            "namespace": self.namespace,
            "name": f"inference_result_{job_name}",
            "facets": {
                "virtual": {
                    "_producer": self.producer,
                    "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/CustomFacet.json",
                    "isVirtual": True
                }
            }
        }
        
        # START event
        start_event, run_id = self.create_lineage_event(
            job_name=job_name,
            event_type="START",
            inputs=[input_data, model],
            outputs=[output_result],
            job_facets=job_facets,
            run_id=run_id
        )
        
        # Send START event
        self.send_lineage_event(start_event)
        
        return run_id
    
    def complete_inference(self,
                         job_name: str,
                         input_data: Dict,
                         model: Dict,
                         run_id: str,
                         inference_metrics: Dict,
                         additional_facets: Optional[Dict] = None) -> None:
        """Complete a model inference job.
        
        Args:
            job_name: Name of the job
            input_data: Input data
            model: Model used for inference
            run_id: Run ID from start event
            inference_metrics: Metrics from the inference
            additional_facets: Additional job facets
        """
        job_facets = additional_facets or {}
        
        # Add inference metrics facet
        job_facets["inferenceMetrics"] = {
            "_producer": self.producer,
            "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/CustomFacet.json",
            "metrics": inference_metrics
        }
        
        # Output is virtual, representing the inference result
        output_result = {
            "namespace": self.namespace,
            "name": f"inference_result_{job_name}",
            "facets": {
                "virtual": {
                    "_producer": self.producer,
                    "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/CustomFacet.json",
                    "isVirtual": True
                }
            }
        }
        
        # COMPLETE event
        complete_event, _ = self.create_lineage_event(
            job_name=job_name,
            event_type="COMPLETE",
            inputs=[input_data, model],
            outputs=[output_result],
            job_facets=job_facets,
            parent_run_id=run_id
        )
        
        # Send COMPLETE event
        self.send_lineage_event(complete_event)


class LineageTracker:
    """Context manager for tracking OpenLineage events."""
    
    def __init__(self, 
                client: OpenLineageClient, 
                job_name: str, 
                inputs: Optional[List[Dict]] = None, 
                outputs: Optional[List[Dict]] = None,
                job_description: Optional[str] = None,
                job_facets: Optional[Dict] = None):
        """Initialize the lineage tracker.
        
        Args:
            client: OpenLineage client
            job_name: Name of the job
            inputs: Input datasets
            outputs: Output datasets
            job_description: Description of the job
            job_facets: Additional job facets
        """
        self.client = client
        self.job_name = job_name
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.job_description = job_description
        self.job_facets = job_facets or {}
        self.run_id = None
        
        # Add documentation facet if description is provided
        if job_description and "documentation" not in self.job_facets:
            self.job_facets["documentation"] = {
                "_producer": self.client.producer,
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DocumentationJobFacet.json",
                "description": job_description
            }
    
    def __enter__(self):
        """Start tracking lineage by sending a START event."""
        # Send START event
        start_event, run_id = self.client.create_lineage_event(
            job_name=self.job_name,
            event_type="START",
            inputs=self.inputs,
            outputs=self.outputs,
            job_facets=self.job_facets
        )
        
        self.client.send_lineage_event(start_event)
        self.run_id = run_id
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Complete tracking lineage by sending a COMPLETE or FAIL event."""
        if exc_type is not None:
            # There was an exception, send FAIL event
            fail_event, _ = self.client.create_lineage_event(
                job_name=self.job_name,
                event_type="FAIL",
                inputs=self.inputs,
                outputs=self.outputs,
                job_facets={
                    **self.job_facets,
                    "errorMessage": {
                        "_producer": self.client.producer,
                        "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/ErrorMessageJobFacet.json",
                        "message": str(exc_val),
                        "programmingLanguage": "PYTHON",
                        "stack": str(exc_tb)
                    }
                },
                parent_run_id=self.run_id
            )
            
            self.client.send_lineage_event(fail_event)
        else:
            # Successful execution, send COMPLETE event
            complete_event, _ = self.client.create_lineage_event(
                job_name=self.job_name,
                event_type="COMPLETE",
                inputs=self.inputs,
                outputs=self.outputs,
                job_facets=self.job_facets,
                parent_run_id=self.run_id
            )
            
            self.client.send_lineage_event(complete_event)
    
    def update_facets(self, facets: Dict):
        """Update job facets during execution.
        
        Args:
            facets: Dictionary of facets to update/add
        """
        self.job_facets.update(facets)
    
    def add_input(self, dataset: Dict):
        """Add an input dataset during execution.
        
        Args:
            dataset: Input dataset to add
        """
        self.inputs.append(dataset)
    
    def add_output(self, dataset: Dict):
        """Add an output dataset during execution.
        
        Args:
            dataset: Output dataset to add
        """
        self.outputs.append(dataset)
        
    def add_job_facet(self, name: str, facet: Dict):
        """Add a job facet with a specific name.
        
        Args:
            name: Name of the facet
            facet: Facet data to add
        """
        if not isinstance(facet, dict):
            facet = {"value": facet}
            
        self.job_facets[name] = {
            "_producer": self.client.producer,
            "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/CustomFacet.json",
            **facet
        }
        
    def add_input_dataset(self, namespace: str, name: str, facets: Optional[Dict] = None):
        """Add an input dataset with namespace and name during execution.
        
        Args:
            namespace: Dataset namespace
            name: Dataset name
            facets: Optional dataset facets
        """
        dataset = {
            "namespace": namespace,
            "name": name
        }
        
        if facets:
            dataset["facets"] = facets
            
        self.inputs.append(dataset)
        
    def add_output_dataset(self, namespace: str, name: str, facets: Optional[Dict] = None):
        """Add an output dataset with namespace and name during execution.
        
        Args:
            namespace: Dataset namespace
            name: Dataset name
            facets: Optional dataset facets
        """
        dataset = {
            "namespace": namespace,
            "name": name
        }
        
        if facets:
            dataset["facets"] = facets
            
        self.outputs.append(dataset)
