#!/usr/bin/env python3
"""
Ray + Iceberg + OpenLineage Setup Verification Script

This script verifies the setup of all necessary components:
1. Python dependencies
2. Ray installation and cluster
3. Marquez API connectivity
4. Virtual environment setup

Run this script to ensure that everything is properly configured.
"""

import sys
import os
import importlib
import subprocess
import requests
import time

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 50)
    print(f" {text}")
    print("=" * 50)

def print_success(text):
    """Print a success message."""
    print(f"✅ {text}")

def print_error(text):
    """Print an error message."""
    print(f"❌ {text}")

def print_warning(text):
    """Print a warning message."""
    print(f"⚠️ {text}")

def print_info(text):
    """Print an info message."""
    print(f"ℹ️ {text}")

def check_virtual_env():
    """Check if running in a virtual environment."""
    print_header("Checking Virtual Environment")
    
    if os.environ.get('VIRTUAL_ENV'):
        venv_name = os.path.basename(os.environ.get('VIRTUAL_ENV'))
        print_success(f"Running in virtual environment: {venv_name}")
        return True
    else:
        print_warning("Not running in a virtual environment")
        print_info("It's recommended to run in a virtual environment.")
        print_info("You can create one with: python3 -m venv ray_lineage_venv")
        print_info("And activate it with: source ray_lineage_venv/bin/activate")
        return False

def check_dependencies():
    """Check if all required dependencies are installed."""
    print_header("Checking Required Dependencies")
    
    required_packages = [
        ("ray", "Ray"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("tensorflow", "TensorFlow"),
        ("openlineage.client", "OpenLineage")
    ]
    
    all_installed = True
    
    for package_name, display_name in required_packages:
        try:
            # Try to import the package
            module = importlib.import_module(package_name)
            
            # Get version if possible
            version = getattr(module, "__version__", "unknown version")
            print_success(f"{display_name} is installed ({version})")
            
        except ImportError:
            print_error(f"{display_name} is not installed")
            all_installed = False
    
    return all_installed

def check_ray_cluster():
    """Check Ray cluster status."""
    print_header("Checking Ray Cluster")
    
    try:
        import ray
        
        # Check if Ray is initialized
        if ray.is_initialized():
            print_success("Ray is already initialized")
            resources = ray.cluster_resources()
            print_info(f"Available CPUs: {resources.get('CPU', 0)}")
            print_info(f"Available GPUs: {resources.get('GPU', 0)}")
            print_info(f"Available memory: {resources.get('memory', 0) / 1e9:.2f} GB")
            
            # Get dashboard URL (handle different versions of Ray)
            try:
                if hasattr(ray, 'get_dashboard_url'):
                    dashboard_url = ray.get_dashboard_url()
                    print_info(f"Ray Dashboard: {dashboard_url}")
                else:
                    print_info("Ray Dashboard should be available at http://localhost:8265")
            except:
                print_info("Ray Dashboard should be available at http://localhost:8265")
                
            return True
        else:
            # Try to initialize Ray
            try:
                print_info("Initializing Ray...")
                ray.init()
                print_success("Ray initialized successfully")
                resources = ray.cluster_resources()
                print_info(f"Available CPUs: {resources.get('CPU', 0)}")
                print_info(f"Available GPUs: {resources.get('GPU', 0)}")
                print_info(f"Available memory: {resources.get('memory', 0) / 1e9:.2f} GB")
                
                # Get dashboard URL (handle different versions of Ray)
                try:
                    if hasattr(ray, 'get_dashboard_url'):
                        dashboard_url = ray.get_dashboard_url()
                        print_info(f"Ray Dashboard: {dashboard_url}")
                    else:
                        print_info("Ray Dashboard should be available at http://localhost:8265")
                except:
                    print_info("Ray Dashboard should be available at http://localhost:8265")
                    
                return True
            except Exception as e:
                print_error(f"Failed to initialize Ray: {e}")
                return False
    except Exception as e:
        print_error(f"Error checking Ray: {e}")
        return False

def check_marquez_connectivity():
    """Check Marquez API connectivity."""
    print_header("Checking Marquez Connectivity")
    
    marquez_url = "http://localhost:5002"
    marquez_web_url = "http://localhost:3000"
    
    # Check if Marquez API is available
    try:
        response = requests.get(f"{marquez_url}/api/v1/namespaces", timeout=5)
        if response.status_code == 200:
            try:
                # Parse the response, which could be different formats
                namespaces_data = response.json()
                
                # Handle different response formats
                if isinstance(namespaces_data, list):
                    # Format: [{'name': 'namespace1', ...}, {'name': 'namespace2', ...}]
                    namespace_count = len(namespaces_data)
                    namespace_names = [ns.get('name', 'unknown') for ns in namespaces_data]
                elif isinstance(namespaces_data, dict) and 'namespaces' in namespaces_data:
                    # Format: {'namespaces': [{'name': 'namespace1', ...}, {'name': 'namespace2', ...}]}
                    namespace_count = len(namespaces_data['namespaces'])
                    namespace_names = [ns.get('name', 'unknown') for ns in namespaces_data['namespaces']]
                else:
                    # If it's a string or other unexpected format
                    namespace_count = 1
                    namespace_names = ["(custom format)"]
                
                print_success(f"Marquez API is available at {marquez_url}")
                print_info(f"Found {namespace_count} namespaces")
                if namespace_count > 0 and namespace_names:
                    print_info(f"Namespaces: {', '.join(str(name) for name in namespace_names)}")
                
                # Also check if the web UI is accessible
                try:
                    web_response = requests.get(marquez_web_url, timeout=2)
                    if web_response.status_code in [200, 302]:
                        print_success(f"Marquez Web UI is available at {marquez_web_url}")
                except:
                    print_warning(f"Marquez Web UI might not be accessible at {marquez_web_url}")
                
                return True
            
            except Exception as e:
                # If we can't parse the response but got a 200
                print_success(f"Marquez API is available at {marquez_url}")
                print_warning(f"Couldn't parse namespaces response: {e}")
                return True
        else:
            print_error(f"Marquez API returned status code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"Failed to connect to Marquez API: {e}")
        print_info(f"Make sure Marquez is running: docker-compose up -d")
        return False

def check_directory_structure():
    """Check if the directory structure is as expected."""
    print_header("Checking Directory Structure")
    
    expected_files = [
        "ray_iceberg_lineage.py",
        "run_demo.sh",
        "setup_env.sh",
        "requirements.txt",
        "docker-compose.yml",
        "README.md"
    ]
    
    expected_dirs = [
        "src",
        "storage"
    ]
    
    all_present = True
    
    for file_name in expected_files:
        if os.path.isfile(file_name):
            print_success(f"Found file: {file_name}")
        else:
            print_warning(f"Missing file: {file_name}")
            all_present = False
    
    for dir_name in expected_dirs:
        if os.path.isdir(dir_name):
            print_success(f"Found directory: {dir_name}")
        else:
            if dir_name == "storage":
                # Storage directory can be created automatically
                print_info(f"Creating directory: {dir_name}")
                os.makedirs(dir_name, exist_ok=True)
                os.makedirs(os.path.join(dir_name, "datasets"), exist_ok=True)
                os.makedirs(os.path.join(dir_name, "models"), exist_ok=True)
                print_success(f"Created directory: {dir_name}")
            else:
                print_warning(f"Missing directory: {dir_name}")
                all_present = False
    
    return all_present

def main():
    """Main verification function."""
    print("\n" + "=" * 50)
    print(" Ray + Iceberg + OpenLineage Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Virtual Environment", check_virtual_env),
        ("Dependencies", check_dependencies),
        ("Ray Cluster", check_ray_cluster),
        ("Marquez Connectivity", check_marquez_connectivity),
        ("Directory Structure", check_directory_structure)
    ]
    
    results = {}
    
    for name, check_func in checks:
        try:
            result = check_func()
            results[name] = result
        except Exception as e:
            print_error(f"Check '{name}' failed with error: {e}")
            results[name] = False
    
    # Print summary
    print_header("Summary")
    
    all_passed = True
    for name, result in results.items():
        if result:
            print_success(f"{name}: Passed")
        else:
            print_error(f"{name}: Failed")
            all_passed = False
    
    if all_passed:
        print("\n✅ All checks passed! Your environment is properly set up.")
        print("\nYou can run the demo with:")
        print("  ./run_demo.sh")
    else:
        print("\n⚠️ Some checks failed. Please address the issues above.")
    
    # If Ray was initialized in this script, shut it down
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
            print_info("Ray has been shut down")
    except:
        pass
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 