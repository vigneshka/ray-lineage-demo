"""
Ray Data Parquet Integration with Lineage Tracking (Simulating Iceberg)
This module provides a simplified Iceberg-like integration using Ray's Parquet methods with OpenLineage tracking.
"""

import os
import ray
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import time
import json

class IcebergManager:
    """Manager for Ray Data's Parquet integration with lineage tracking (simulating Iceberg behavior)."""
    
    def __init__(self, 
                warehouse_path: str, 
                namespace: str = "ray_demo",
                catalog_name: str = "demo_catalog"):
        """Initialize the simplified Iceberg-like manager.
        
        Args:
            warehouse_path: Path to warehouse directory
            namespace: Database namespace 
            catalog_name: Name for the catalog
        """
        self.warehouse_path = os.path.abspath(warehouse_path)
        self.namespace = namespace
        self.catalog_name = catalog_name
        
        # Ensure warehouse directory exists
        os.makedirs(self.warehouse_path, exist_ok=True)
        
        # Create namespace directory
        self.namespace_dir = os.path.join(self.warehouse_path, self.namespace)
        os.makedirs(self.namespace_dir, exist_ok=True)
        
        # Create a metadata directory to track table versions
        self.metadata_dir = os.path.join(self.warehouse_path, "_metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        print(f"✅ Initialized simulated Iceberg catalog '{catalog_name}' at {warehouse_path}")
    
    def get_table_path(self, table_name: str) -> str:
        """Get the path for a table.
        
        Args:
            table_name: Simple table name
            
        Returns:
            Path to the table directory
        """
        return os.path.join(self.namespace_dir, table_name)
    
    def get_full_table_identifier(self, table_name: str) -> str:
        """Get fully qualified table identifier.
        
        Args:
            table_name: Simple table name
            
        Returns:
            Fully qualified table identifier: {namespace}.{table_name}
        """
        return f"{self.namespace}.{table_name}"
    
    def create_table(self,
                    table_name: str,
                    df: pd.DataFrame,
                    metadata: Optional[Dict] = None,
                    lineage_tracker = None) -> Dict[str, Any]:
        """Create a table from a pandas DataFrame with lineage tracking.
        
        Args:
            table_name: Name of the table (without namespace)
            df: Pandas DataFrame
            metadata: Optional metadata to include
            lineage_tracker: Optional LineageTracker context manager
            
        Returns:
            Dictionary with metadata about the created table
        """
        # Convert pandas DataFrame to Ray Dataset
        ds = ray.data.from_pandas(df)
        
        # Use write_dataset method to handle the actual writing
        return self.write_dataset(
            ds=ds,
            table_name=table_name,
            mode="overwrite",
            metadata=metadata,
            lineage_tracker=lineage_tracker
        )
    
    def write_dataset(self, 
                     ds: ray.data.Dataset, 
                     table_name: str,
                     mode: str = "overwrite",
                     metadata: Optional[Dict] = None,
                     lineage_tracker = None) -> Dict[str, Any]:
        """Write Ray Dataset to the table with lineage tracking.
        
        Args:
            ds: Ray Dataset to write
            table_name: Name of the table (without namespace)
            mode: Write mode ("overwrite" or "append")
            metadata: Optional metadata to include
            lineage_tracker: Optional LineageTracker context manager
            
        Returns:
            Dictionary with metadata about the written table
        """
        # Get full table identifier and path
        table_identifier = self.get_full_table_identifier(table_name)
        table_path = self.get_table_path(table_name)
        
        # Create table directory if it doesn't exist
        os.makedirs(table_path, exist_ok=True)
        
        # Generate a new snapshot ID (using timestamp)
        snapshot_id = int(time.time())
        
        # Prepare metadata for lineage tracking
        table_metadata = {
            "table_identifier": table_identifier,
            "snapshot_id": snapshot_id,
            "count": ds.count(),
            "num_blocks": ds.num_blocks(),
            "size_bytes": ds.size_bytes(),
            "path": table_path
        }
        
        # Include any user-provided metadata
        if metadata:
            table_metadata.update(metadata)
        
        # Track as output in lineage if tracker provided
        if lineage_tracker:
            # Sample schema from Ray dataset
            sample = ds.take(1)
            schema = []
            if sample:
                schema = [{
                    "name": key,
                    "type": type(value).__name__
                } for key, value in sample[0].items()]
            
            # Add as job facet to lineage tracker
            lineage_tracker.add_job_facet("iceberg_write_info", {
                "table_identifier": table_identifier,
                "mode": mode,
                "schema": schema,
                "count": ds.count(),
                "catalog": self.catalog_name,
                "metadata": metadata
            })
            
            # Add as output dataset
            output_dataset = {
                "namespace": self.namespace,
                "name": f"iceberg://{table_name}",
                "facets": {
                    "schema": {
                        "_producer": lineage_tracker.client.producer,
                        "_schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json#/definitions/SchemaDatasetFacet",
                        "fields": schema
                    },
                    "dataSource": {
                        "_producer": lineage_tracker.client.producer,
                        "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DatasourceDatasetFacet.json",
                        "name": table_name,
                        "uri": f"iceberg://{self.catalog_name}/{table_identifier}"
                    }
                }
            }
            
            # Add metadata as custom facets
            if metadata:
                output_dataset["facets"]["custom"] = {
                    "_producer": lineage_tracker.client.producer,
                    "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/CustomFacet.json",
                    "metadata": metadata
                }
            
            lineage_tracker.add_output(output_dataset)
        
        # Write dataset to parquet file
        try:
            # Create snapshot directory
            snapshot_dir = os.path.join(table_path, f"snapshot-{snapshot_id}")
            os.makedirs(snapshot_dir, exist_ok=True)
            
            # Write the dataset to parquet
            ds.write_parquet(
                path=snapshot_dir
            )
            
            # Write metadata file
            metadata_file = os.path.join(self.metadata_dir, f"{table_name}-{snapshot_id}.json")
            with open(metadata_file, "w") as f:
                json.dump(table_metadata, f, indent=2)
            
            # Write latest pointer file
            latest_file = os.path.join(table_path, "LATEST")
            with open(latest_file, "w") as f:
                f.write(str(snapshot_id))
            
            print(f"✅ Successfully wrote to simulated Iceberg table: {table_identifier}")
            
            # Update metadata with success status
            table_metadata["status"] = "success"
            return table_metadata
            
        except Exception as e:
            print(f"❌ Failed to write to simulated Iceberg table: {e}")
            if lineage_tracker:
                lineage_tracker.add_job_facet("error", {
                    "message": str(e),
                    "type": type(e).__name__
                })
            
            # Update metadata with error
            table_metadata["status"] = "error"
            table_metadata["error"] = str(e)
            return table_metadata
    
    def read_dataset(self, 
                    table_name: str,
                    row_filter = None,
                    selected_fields: Optional[List[str]] = None,
                    snapshot_id: Optional[int] = None,
                    lineage_tracker = None) -> ray.data.Dataset:
        """Read from table with lineage tracking.
        
        Args:
            table_name: Name of the table (without namespace)
            row_filter: Filter expression (not supported in simplified version)
            selected_fields: Optional list of fields to select
            snapshot_id: Optional snapshot ID for time travel
            lineage_tracker: Optional LineageTracker context manager
            
        Returns:
            Ray Dataset loaded from the table
        """
        # Get full table identifier and path
        table_identifier = self.get_full_table_identifier(table_name)
        table_path = self.get_table_path(table_name)
        
        # If snapshot_id is not provided, get the latest
        if snapshot_id is None:
            try:
                latest_file = os.path.join(table_path, "LATEST")
                if os.path.exists(latest_file):
                    with open(latest_file, "r") as f:
                        snapshot_id = int(f.read().strip())
                else:
                    raise ValueError(f"No LATEST file found for table {table_name}")
            except Exception as e:
                raise ValueError(f"Failed to get latest snapshot ID for table {table_name}: {e}")
        
        # Prepare snapshot path
        snapshot_dir = os.path.join(table_path, f"snapshot-{snapshot_id}")
        if not os.path.exists(snapshot_dir):
            raise ValueError(f"Snapshot {snapshot_id} not found for table {table_name}")
        
        # Track as input in lineage if tracker provided
        if lineage_tracker:
            # Add job facet with query details
            lineage_tracker.add_job_facet("iceberg_read_info", {
                "table_identifier": table_identifier,
                "filter": str(row_filter) if row_filter else None,
                "selected_fields": selected_fields if selected_fields else "all fields",
                "snapshot_id": snapshot_id,
                "catalog": self.catalog_name
            })
            
            # Add a simplified input dataset to lineage
            lineage_tracker.add_input({
                "namespace": self.namespace,
                "name": f"iceberg://{table_name}",
                "facets": {
                    "dataSource": {
                        "_producer": lineage_tracker.client.producer,
                        "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DatasourceDatasetFacet.json",
                        "name": table_name,
                        "uri": f"iceberg://{self.catalog_name}/{table_identifier}"
                    }
                }
            })
        
        # Read from parquet file
        try:
            ds = ray.data.read_parquet(snapshot_dir)
            
            # Apply column selection if specified
            if selected_fields:
                ds = ds.select_columns(selected_fields)
            
            # Note: row_filter is not supported in this simplified version
            if row_filter:
                print(f"⚠️ Row filtering is not supported in the simplified Iceberg manager")
            
            return ds
            
        except Exception as e:
            print(f"❌ Failed to read from simulated Iceberg table: {e}")
            if lineage_tracker:
                lineage_tracker.add_job_facet("error", {
                    "message": str(e),
                    "type": type(e).__name__
                })
            raise
    
    def load_table_as_pandas(self,
                           table_name: str,
                           row_filter = None,
                           selected_fields: Optional[List[str]] = None,
                           snapshot_id: Optional[int] = None,
                           lineage_tracker = None) -> pd.DataFrame:
        """Read from table as pandas DataFrame with lineage tracking.
        
        Args:
            table_name: Name of the table (without namespace)
            row_filter: Filter expression (not supported in simplified version)
            selected_fields: Optional list of fields to select
            snapshot_id: Optional snapshot ID for time travel
            lineage_tracker: Optional LineageTracker context manager
            
        Returns:
            pandas DataFrame loaded from the table
        """
        # Read as Ray Dataset first
        ds = self.read_dataset(
            table_name=table_name,
            row_filter=row_filter,
            selected_fields=selected_fields,
            snapshot_id=snapshot_id,
            lineage_tracker=lineage_tracker
        )
        
        # Convert to pandas
        return ds.to_pandas()
    
    def list_tables(self) -> List[str]:
        """List all tables in the namespace.
        
        Returns:
            List of table names without namespace prefix
        """
        if not os.path.exists(self.namespace_dir):
            return []
        
        # List all directories in the namespace directory
        tables = []
        for item in os.listdir(self.namespace_dir):
            item_path = os.path.join(self.namespace_dir, item)
            if os.path.isdir(item_path):
                tables.append(item)
        
        return tables 