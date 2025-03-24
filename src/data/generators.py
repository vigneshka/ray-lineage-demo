"""
Data Generator Module for Ray + Iceberg + OpenLineage Demo
This module provides synthetic data generation capabilities.
"""

import ray
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Any


class DataGenerator:
    """Generate synthetic datasets for demonstration purposes."""
    
    @staticmethod
    def generate_customer_dataset(n_rows: int = 1000, 
                                 version: str = "1.0.0",
                                 random_seed: int = 42,
                                 noise_level: float = 0.5,
                                 as_ray_dataset: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """Generate a synthetic customer dataset for churn prediction.
        
        Args:
            n_rows: Number of rows
            version: Version string
            random_seed: Random seed for reproducibility
            noise_level: Amount of noise to add
            as_ray_dataset: Whether to return as Ray Dataset (True) or pandas DataFrame (False)
            
        Returns:
            Tuple of (Dataset, metadata)
        """
        np.random.seed(random_seed)
        
        # Generate features
        age = np.random.normal(45, 15, n_rows).astype(int)
        age = np.clip(age, 18, 95)  # Clip age to reasonable range
        
        tenure = np.random.gamma(shape=2.0, scale=15, size=n_rows).astype(int)
        tenure = np.clip(tenure, 0, 100)  # Clip tenure to reasonable range
        
        monthly_charges = 50 + 50 * np.random.beta(2, 5, n_rows)
        total_charges = monthly_charges * tenure * (0.9 + 0.2 * np.random.random(n_rows))
        
        # Different versions have different characteristics
        if version == "1.0.0":
            # Basic version
            contract_types = np.random.choice(
                ['Month-to-month', 'One year', 'Two year'], 
                p=[0.6, 0.3, 0.1], 
                size=n_rows
            )
            
            # Simple churn relationship
            churn_probability = 0.3 - 0.003 * tenure + 0.003 * monthly_charges
            churn_probability = np.clip(churn_probability, 0.05, 0.95)
            
            # Create the DataFrame
            data = {
                'CustomerId': [f'CUST-{i:06d}' for i in range(1, n_rows + 1)],
                'Age': age,
                'Tenure': tenure,
                'ContractType': contract_types,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges,
            }
            
        elif version == "2.0.0":
            # More contract types with different distribution
            contract_types = np.random.choice(
                ['Month-to-month', 'One year', 'Two year', 'Custom'], 
                p=[0.5, 0.25, 0.15, 0.1], 
                size=n_rows
            )
            
            # Additional variables in v2
            has_phone_service = np.random.choice([True, False], p=[0.9, 0.1], size=n_rows)
            has_internet_service = np.random.choice([True, False], p=[0.8, 0.2], size=n_rows)
            
            # More complex churn relationship
            churn_probability = (
                0.25 - 0.005 * tenure + 
                0.004 * monthly_charges + 
                0.2 * (contract_types == 'Month-to-month') -
                0.1 * (contract_types == 'Two year') -
                0.05 * has_phone_service +
                0.1 * (has_internet_service)
            )
            churn_probability = np.clip(churn_probability, 0.05, 0.95)
            
            # Create the DataFrame with additional columns
            data = {
                'CustomerId': [f'CUST-{i:06d}' for i in range(1, n_rows + 1)],
                'Age': age,
                'Tenure': tenure,
                'ContractType': contract_types,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges,
                'HasPhoneService': has_phone_service,
                'HasInternetService': has_internet_service
            }
            
        else:  # version 3.0.0 or others
            # Even more complex version
            contract_types = np.random.choice(
                ['Month-to-month', 'One year', 'Two year', 'Custom', 'Flex'], 
                p=[0.4, 0.2, 0.15, 0.15, 0.1], 
                size=n_rows
            )
            
            # Additional variables in v3
            internet_service = np.random.choice(
                ['DSL', 'Fiber optic', 'No'], 
                p=[0.3, 0.6, 0.1], 
                size=n_rows
            )
            
            payment_method = np.random.choice(
                ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], 
                p=[0.4, 0.1, 0.25, 0.25], 
                size=n_rows
            )
            
            # Customer segment based on value and risk
            customer_value = np.random.beta(2, 5, n_rows) * 100
            risk_score = np.random.beta(1.5, 8, n_rows) * 100
            
            # Complex churn relationship
            churn_probability = (
                0.2 - 0.004 * tenure + 
                0.005 * monthly_charges + 
                0.25 * (contract_types == 'Month-to-month') -
                0.15 * (contract_types == 'Two year') +
                0.1 * (internet_service == 'Fiber optic') -
                0.05 * (internet_service == 'No') +
                0.1 * (payment_method == 'Electronic check') +
                0.001 * customer_value -
                0.002 * risk_score
            )
            churn_probability = np.clip(churn_probability, 0.05, 0.95)
            
            # Create the DataFrame with all columns
            data = {
                'CustomerId': [f'CUST-{i:06d}' for i in range(1, n_rows + 1)],
                'Age': age,
                'Tenure': tenure,
                'ContractType': contract_types,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges,
                'InternetService': internet_service,
                'PaymentMethod': payment_method,
                'CustomerValue': customer_value,
                'RiskScore': risk_score
            }
        
        # Add noise to the churn probability
        churn_probability += noise_level * np.random.normal(0, 0.1, n_rows)
        churn_probability = np.clip(churn_probability, 0.05, 0.95)
        
        # Generate the target variable
        churned = np.random.random(n_rows) < churn_probability
        
        # Add the target variable to all versions
        data['Churned'] = churned
        
        # Create pandas DataFrame
        df = pd.DataFrame(data)
        
        # Create metadata
        metadata = {
            "name": "customer_churn",
            "version": version,
            "rows": n_rows,
            "features": list(df.columns),
            "target": "Churned",
            "generation_params": {
                "random_seed": random_seed,
                "noise_level": noise_level
            },
            "description": f"Synthetic customer churn dataset v{version}"
        }
        
        # Convert to Ray Dataset if requested
        if as_ray_dataset:
            ds = ray.data.from_pandas(df)
            return ds, metadata
        else:
            return df, metadata
    
    @staticmethod
    def generate_credit_dataset(n_rows: int = 1000, 
                              version: str = "1.0.0",
                              random_seed: int = 42,
                              noise_level: float = 0.5,
                              as_ray_dataset: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """Generate a synthetic credit card dataset for fraud detection.
        
        Args:
            n_rows: Number of rows
            version: Version string
            random_seed: Random seed for reproducibility
            noise_level: Amount of noise to add
            as_ray_dataset: Whether to return as Ray Dataset (True) or pandas DataFrame (False)
            
        Returns:
            Tuple of (Dataset, metadata)
        """
        np.random.seed(random_seed)
        
        # Generate features
        transaction_amount = np.exp(np.random.normal(4, 1.5, n_rows))  # Log-normal distribution
        transaction_amount = np.clip(transaction_amount, 1, 5000)
        
        # Different versions have different characteristics and fields
        if version == "1.0.0":
            # Basic version with limited features
            hour_of_day = np.random.randint(0, 24, n_rows)
            day_of_week = np.random.randint(0, 7, n_rows)
            
            # Simple fraud indicators
            distance_from_home = np.random.exponential(30, n_rows)
            distance_from_home = np.clip(distance_from_home, 0, 500)
            
            # Simple fraud model
            fraud_probability = (
                0.01 +  # Base fraud rate
                0.001 * transaction_amount / 100 +  # Higher amounts increase fraud risk
                0.01 * (hour_of_day >= 1) * (hour_of_day <= 5) +  # Late night transactions
                0.02 * (distance_from_home > 100)  # Unusual location
            )
            
            # Create the DataFrame
            data = {
                'TransactionId': [f'TX-{i:08d}' for i in range(1, n_rows + 1)],
                'Amount': transaction_amount,
                'HourOfDay': hour_of_day,
                'DayOfWeek': day_of_week,
                'DistanceFromHome': distance_from_home
            }
            
        elif version == "2.0.0":
            # More features in v2
            hour_of_day = np.random.randint(0, 24, n_rows)
            day_of_week = np.random.randint(0, 7, n_rows)
            
            distance_from_home = np.random.exponential(30, n_rows)
            distance_from_home = np.clip(distance_from_home, 0, 500)
            
            distance_from_last_transaction = np.random.exponential(20, n_rows)
            distance_from_last_transaction = np.clip(distance_from_last_transaction, 0, 1000)
            
            ratio_to_median_purchase_price = np.random.exponential(1, n_rows)
            ratio_to_median_purchase_price = np.clip(ratio_to_median_purchase_price, 0.1, 10)
            
            merchant_categories = np.random.choice(
                ['grocery', 'restaurant', 'gas', 'online', 'retail', 'travel', 'other'],
                p=[0.2, 0.2, 0.15, 0.2, 0.1, 0.05, 0.1],
                size=n_rows
            )
            
            # More complex fraud model
            fraud_probability = (
                0.005 +  # Lower base fraud rate
                0.0008 * transaction_amount / 100 +  # Higher amounts
                0.01 * (hour_of_day >= 1) * (hour_of_day <= 5) +  # Late night
                0.015 * (distance_from_home > 100) +  # Unusual location
                0.02 * (distance_from_last_transaction > 200) +  # Unusual travel
                0.03 * (ratio_to_median_purchase_price > 5) +  # Unusual purchase amount
                0.02 * (merchant_categories == 'online')  # Online purchases higher risk
            )
            
            # Create the DataFrame with additional columns
            data = {
                'TransactionId': [f'TX-{i:08d}' for i in range(1, n_rows + 1)],
                'Amount': transaction_amount,
                'HourOfDay': hour_of_day,
                'DayOfWeek': day_of_week,
                'DistanceFromHome': distance_from_home,
                'DistanceFromLastTransaction': distance_from_last_transaction,
                'RatioToMedianPurchasePrice': ratio_to_median_purchase_price,
                'MerchantCategory': merchant_categories
            }
            
        else:  # version 3.0.0 or others
            # Most comprehensive feature set in v3
            timestamp = np.array([
                pd.Timestamp('2023-01-01') + pd.Timedelta(seconds=i)
                for i in np.random.randint(0, 60*60*24*365, n_rows)
            ])
            
            hour_of_day = np.array([ts.hour for ts in timestamp])
            day_of_week = np.array([ts.dayofweek for ts in timestamp])
            month = np.array([ts.month for ts in timestamp])
            
            distance_from_home = np.random.exponential(30, n_rows)
            distance_from_home = np.clip(distance_from_home, 0, 500)
            
            distance_from_last_transaction = np.random.exponential(20, n_rows)
            distance_from_last_transaction = np.clip(distance_from_last_transaction, 0, 1000)
            
            ratio_to_median_purchase_price = np.random.exponential(1, n_rows)
            ratio_to_median_purchase_price = np.clip(ratio_to_median_purchase_price, 0.1, 10)
            
            merchant_categories = np.random.choice(
                ['grocery', 'restaurant', 'gas', 'online', 'retail', 'travel', 'other'],
                p=[0.2, 0.2, 0.15, 0.2, 0.1, 0.05, 0.1],
                size=n_rows
            )
            
            # New fields in v3
            repeat_retailer = np.random.choice([True, False], p=[0.7, 0.3], size=n_rows)
            used_chip = np.random.choice([True, False], p=[0.8, 0.2], size=n_rows)
            used_pin_number = np.random.choice([True, False], p=[0.6, 0.4], size=n_rows)
            online_order = np.random.choice([True, False], p=[0.4, 0.6], size=n_rows)
            
            fraud_signals = np.random.beta(1, 30, n_rows)  # Rare high values indicate fraud
            
            # Complex fraud model
            fraud_probability = (
                0.003 +  # Even lower base fraud rate
                0.0005 * transaction_amount / 100 +  # Higher amounts 
                0.008 * (hour_of_day >= 1) * (hour_of_day <= 5) +  # Late night
                0.01 * (distance_from_home > 100) +  # Unusual location
                0.015 * (distance_from_last_transaction > 200) +  # Unusual travel
                0.02 * (ratio_to_median_purchase_price > 5) +  # Unusual purchase amount
                0.01 * (merchant_categories == 'online') +  # Online purchases
                -0.01 * repeat_retailer +  # Familiar merchants reduce risk
                -0.01 * used_chip +  # Chip use reduces risk
                -0.005 * used_pin_number +  # PIN use reduces risk
                0.01 * online_order +  # Online orders increase risk
                0.5 * fraud_signals  # Dedicated fraud signals (from detection systems)
            )
            
            # Create the DataFrame with all columns
            data = {
                'TransactionId': [f'TX-{i:08d}' for i in range(1, n_rows + 1)],
                'Timestamp': timestamp,
                'Amount': transaction_amount,
                'HourOfDay': hour_of_day,
                'DayOfWeek': day_of_week,
                'Month': month,
                'DistanceFromHome': distance_from_home,
                'DistanceFromLastTransaction': distance_from_last_transaction,
                'RatioToMedianPurchasePrice': ratio_to_median_purchase_price,
                'MerchantCategory': merchant_categories,
                'RepeatRetailer': repeat_retailer,
                'UsedChip': used_chip,
                'UsedPinNumber': used_pin_number,
                'OnlineOrder': online_order,
                'FraudSignals': fraud_signals
            }
        
        # Add noise to the fraud probability
        fraud_probability += noise_level * np.random.normal(0, 0.01, n_rows)
        fraud_probability = np.clip(fraud_probability, 0.001, 0.99)
        
        # Generate the target variable
        is_fraud = np.random.random(n_rows) < fraud_probability
        
        # Add the target variable to all versions
        data['IsFraud'] = is_fraud
        
        # Create pandas DataFrame
        df = pd.DataFrame(data)
        
        # Create metadata
        metadata = {
            "name": "credit_fraud",
            "version": version,
            "rows": n_rows,
            "features": list(df.columns),
            "target": "IsFraud",
            "generation_params": {
                "random_seed": random_seed,
                "noise_level": noise_level
            },
            "description": f"Synthetic credit card fraud dataset v{version}"
        }
        
        # Convert to Ray Dataset if requested
        if as_ray_dataset:
            ds = ray.data.from_pandas(df)
            return ds, metadata
        else:
            return df, metadata
