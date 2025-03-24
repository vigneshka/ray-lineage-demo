"""
Model Trainer Module for Ray + Iceberg + OpenLineage Demo
This module provides model training capabilities using Ray Train.
"""

import ray
from ray import train
import ray.train.tensorflow

# Fix for sklearn integration - don't explicitly import ray.train.sklearn
# Instead use sklearn models directly
import pandas as pd
import numpy as np
import pickle
import os
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


class ModelTrainer:
    """Base class for model training with Ray Train."""
    
    def __init__(self, 
                model_name: str,
                model_version: str = "1.0.0",
                random_state: int = 42,
                model_dir: str = "./models",
                training_split: float = 0.8):
        """Initialize model trainer.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            random_state: Random seed for reproducibility
            model_dir: Directory to save models
            training_split: Proportion of data to use for training
        """
        self.model_name = model_name
        self.model_version = model_version
        self.random_state = random_state
        self.model_dir = model_dir
        self.training_split = training_split
        self.model = None
        self.preprocessor = None
        self.hyperparameters = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
    def prepare_data(self, 
                    dataset: ray.data.Dataset, 
                    target_column: str) -> Tuple[ray.data.Dataset, ray.data.Dataset]:
        """Prepare data for training.
        
        Args:
            dataset: Input Ray Dataset
            target_column: Name of target column
            
        Returns:
            Train and test datasets
        """
        # Split dataset into train and test
        train_dataset, test_dataset = dataset.train_test_split(
            test_size=(1-self.training_split), 
            seed=self.random_state
        )
        
        return train_dataset, test_dataset
    
    def _get_model_path(self) -> str:
        """Get path to save model."""
        return os.path.join(
            self.model_dir, 
            f"{self.model_name}_{self.model_version}.pkl"
        )
    
    def _get_metadata_path(self) -> str:
        """Get path to save model metadata."""
        return os.path.join(
            self.model_dir, 
            f"{self.model_name}_{self.model_version}_metadata.json"
        )
        
    def save_model(self, model: Any, metadata: Dict[str, Any]) -> None:
        """Save model and metadata.
        
        Args:
            model: Trained model object
            metadata: Model metadata
        """
        model_path = self._get_model_path()
        metadata_path = self._get_metadata_path()
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {metadata_path}")
        
    def load_model(self) -> Tuple[Any, Dict[str, Any]]:
        """Load model and metadata.
        
        Returns:
            Tuple of (model, metadata)
        """
        model_path = self._get_model_path()
        metadata_path = self._get_metadata_path()
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        return model, metadata
    
    def train(self, 
             dataset: ray.data.Dataset, 
             target_column: str, 
             feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train model.
        
        This should be implemented by subclasses.
        
        Args:
            dataset: Input Ray Dataset
            target_column: Name of target column
            feature_columns: List of feature columns to use
            
        Returns:
            Dictionary with training metrics
        """
        raise NotImplementedError("Subclasses must implement train()")
    
    def evaluate(self, 
                model: Any,
                test_dataset: ray.data.Dataset,
                target_column: str,
                feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate model on test dataset.
        
        This should be implemented by subclasses.
        
        Args:
            model: Trained model
            test_dataset: Test Ray Dataset
            target_column: Name of target column
            feature_columns: List of feature columns to use
            
        Returns:
            Dictionary with evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement evaluate()")


class SklearnModelTrainer(ModelTrainer):
    """Trainer for scikit-learn models with Ray Train."""
    
    def __init__(self, 
                model_name: str,
                model_type: str = "random_forest",
                model_version: str = "1.0.0",
                random_state: int = 42,
                model_dir: str = "./models",
                training_split: float = 0.8,
                hyperparameters: Optional[Dict[str, Any]] = None):
        """Initialize scikit-learn model trainer.
        
        Args:
            model_name: Name of the model
            model_type: Type of model ('random_forest', 'gradient_boosting', 'logistic_regression')
            model_version: Version of the model
            random_state: Random seed for reproducibility
            model_dir: Directory to save models
            training_split: Proportion of data to use for training
            hyperparameters: Model hyperparameters
        """
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            random_state=random_state,
            model_dir=model_dir,
            training_split=training_split
        )
        
        self.model_type = model_type
        self.hyperparameters = hyperparameters or {}
        
    def _create_model(self) -> Any:
        """Create model based on model type."""
        if self.model_type == "random_forest":
            default_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": self.random_state
            }
            params = {**default_params, **self.hyperparameters}
            return RandomForestClassifier(**params)
            
        elif self.model_type == "gradient_boosting":
            default_params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": self.random_state
            }
            params = {**default_params, **self.hyperparameters}
            return GradientBoostingClassifier(**params)
            
        elif self.model_type == "logistic_regression":
            default_params = {
                "C": 1.0,
                "max_iter": 100,
                "random_state": self.random_state
            }
            params = {**default_params, **self.hyperparameters}
            return LogisticRegression(**params)
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _prepare_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Create preprocessor for features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            scikit-learn ColumnTransformer
        """
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor
        
    def _create_training_pipeline(self) -> Pipeline:
        """Create scikit-learn pipeline with preprocessor and model."""
        model = self._create_model()
        return Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', model)
        ])
    
    def train(self,
             dataset: ray.data.Dataset,
             target_column: str,
             feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train scikit-learn model directly without Ray Train (simplified).
        
        Args:
            dataset: Input Ray Dataset
            target_column: Name of target column
            feature_columns: List of feature columns to use
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare data
        train_dataset, test_dataset = self.prepare_data(dataset, target_column)
        
        # Convert to pandas for preprocessing 
        train_df = train_dataset.to_pandas()
        
        # Filter columns if specified
        if feature_columns is not None:
            X_train = train_df[feature_columns]
        else:
            X_train = train_df.drop(columns=[target_column])
            feature_columns = X_train.columns.tolist()
            
        y_train = train_df[target_column]
        
        # Create preprocessor and fit it
        self.preprocessor = self._prepare_preprocessor(X_train)
        self.preprocessor.fit(X_train)
        
        # Create and train pipeline
        pipeline = self._create_training_pipeline()
        
        # Train the model directly 
        pipeline.fit(X_train, y_train)
        
        # Store the trained pipeline
        trained_pipeline = pipeline
        
        # Extract the model from the pipeline
        self.model = trained_pipeline.named_steps['model']
        
        # Evaluate on test set
        metrics = self.evaluate(
            model=trained_pipeline,
            test_dataset=test_dataset,
            target_column=target_column,
            feature_columns=feature_columns
        )
        
        # Add training information to metrics
        model_info = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "hyperparameters": self.hyperparameters,
            "training_samples": len(X_train),
            "test_samples": test_dataset.count(),
        }
        
        metrics.update({
            "model_info": model_info
        })
        
        # Save model
        self.save_model(trained_pipeline, metrics)
        
        return metrics
    
    def evaluate(self,
                model: Any,
                test_dataset: ray.data.Dataset,
                target_column: str,
                feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate scikit-learn model on test dataset.
        
        Args:
            model: Trained model (can be pipeline or model)
            test_dataset: Test Ray Dataset
            target_column: Name of target column
            feature_columns: List of feature columns to use
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Convert to pandas
        test_df = test_dataset.to_pandas()
        
        # Filter columns if specified
        if feature_columns is not None:
            X_test = test_df[feature_columns]
        else:
            X_test = test_df.drop(columns=[target_column])
            
        y_test = test_df[target_column]
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            has_proba = True
        except (AttributeError, IndexError):
            has_proba = False
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }
        
        # Add AUC if probabilities are available
        if has_proba:
            metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
            
        # Add confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        
        return metrics
    
    def predict(self, 
               data: Union[pd.DataFrame, ray.data.Dataset],
               feature_columns: Optional[List[str]] = None) -> np.ndarray:
        """Make predictions with trained model.
        
        Args:
            data: Input data
            feature_columns: List of feature columns to use
            
        Returns:
            Predictions
        """
        # Load model if not already loaded
        if self.model is None:
            model, _ = self.load_model()
        else:
            model = self.model
            
        # Convert Ray Dataset to pandas if needed
        if isinstance(data, ray.data.Dataset):
            data = data.to_pandas()
            
        # Filter columns if specified
        if feature_columns is not None:
            data = data[feature_columns]
            
        # Make predictions
        return model.predict(data)
    
    def predict_proba(self, 
                     data: Union[pd.DataFrame, ray.data.Dataset],
                     feature_columns: Optional[List[str]] = None) -> np.ndarray:
        """Make probability predictions with trained model.
        
        Args:
            data: Input data
            feature_columns: List of feature columns to use
            
        Returns:
            Probability predictions
        """
        # Load model if not already loaded
        if self.model is None:
            model, _ = self.load_model()
        else:
            model = self.model
            
        # Convert Ray Dataset to pandas if needed
        if isinstance(data, ray.data.Dataset):
            data = data.to_pandas()
            
        # Filter columns if specified
        if feature_columns is not None:
            data = data[feature_columns]
            
        # Make predictions
        return model.predict_proba(data)[:, 1]
