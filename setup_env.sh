#!/bin/bash
# Ray + Iceberg + OpenLineage Demo Environment Setup

# Colors for pretty output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BLUE}${BOLD}Ray + Iceberg + OpenLineage Environment Setup${NC}"
echo "=========================================================="

# Check Python version
if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"
else
    echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
    echo "Please install Python 3.8 or higher before proceeding."
    exit 1
fi

# Default virtual environment name
VENV_NAME="ray_lineage_venv"

# Ask for virtual environment name
read -p "Virtual environment name [$VENV_NAME]: " user_venv
VENV_NAME=${user_venv:-$VENV_NAME}

# Check if virtual environment already exists
if [ -d "$VENV_NAME" ]; then
    echo -e "${YELLOW}Virtual environment '$VENV_NAME' already exists.${NC}"
    read -p "Delete and recreate? (y/n): " recreate
    if [ "$recreate" = "y" ] || [ "$recreate" = "Y" ]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_NAME"
    else
        echo "Using existing virtual environment."
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_NAME" ]; then
    echo -e "${YELLOW}Creating new virtual environment: $VENV_NAME${NC}"
    python3 -m venv "$VENV_NAME"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source "$VENV_NAME/bin/activate"
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate virtual environment.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Virtual environment '$VENV_NAME' activated${NC}"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo -e "${RED}Warning: Failed to upgrade pip. Continuing anyway...${NC}"
fi

# Install requirements
echo -e "${YELLOW}Installing dependencies from requirements.txt...${NC}"
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install all dependencies.${NC}"
    echo -e "${YELLOW}You may need to install some system dependencies or resolve conflicts.${NC}"
    echo "Continue with setup, but the demo might not work correctly."
else
    echo -e "${GREEN}✓ All dependencies installed successfully${NC}"
fi

# Verify TensorFlow installation
echo -e "${YELLOW}Verifying TensorFlow installation...${NC}"
python3 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}⚠️ TensorFlow installation failed or has issues.${NC}"
    echo "This might cause problems when running the demo."
    echo "You can try installing TensorFlow separately:"
    echo -e "${BLUE}pip install tensorflow>=2.10.0${NC}"
else
    echo -e "${GREEN}✓ TensorFlow verified${NC}"
fi

# Check Docker and Docker Compose
echo -e "${YELLOW}Checking Docker installation...${NC}"
if command -v docker >/dev/null 2>&1; then
    DOCKER_VERSION=$(docker --version)
    echo -e "${GREEN}✓ $DOCKER_VERSION${NC}"
else
    echo -e "${RED}⚠️ Docker is not installed or not in PATH.${NC}"
    echo "Docker is required to run Marquez for OpenLineage tracking."
    echo "Please install Docker before running the demo."
fi

# Check Docker Compose
echo -e "${YELLOW}Checking Docker Compose installation...${NC}"
if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_VERSION=$(docker-compose --version)
    echo -e "${GREEN}✓ $COMPOSE_VERSION${NC}"
elif command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    COMPOSE_VERSION=$(docker compose version)
    echo -e "${GREEN}✓ Docker Compose plugin detected${NC}"
else
    echo -e "${RED}⚠️ Docker Compose is not installed.${NC}"
    echo "Docker Compose is required to run Marquez for OpenLineage tracking."
    echo "Please install Docker Compose before running the demo."
fi

# Prompt to start Marquez
echo -e "\n${YELLOW}Do you want to start Marquez containers now? (y/n):${NC} "
read start_marquez
if [ "$start_marquez" = "y" ] || [ "$start_marquez" = "Y" ]; then
    echo -e "${YELLOW}Starting Marquez components with Docker Compose...${NC}"
    if command -v docker-compose >/dev/null 2>&1; then
        docker-compose up -d
    else
        docker compose up -d
    fi
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to start Marquez containers.${NC}"
    else
        echo -e "${GREEN}✓ Marquez started successfully${NC}"
        echo "Marquez Web UI: http://localhost:3000"
        echo "Marquez API: http://localhost:5002"
    fi
fi

# Display summary
echo -e "\n${GREEN}${BOLD}Setup Complete!${NC}"
echo -e "Virtual Environment: ${BLUE}$VENV_NAME${NC}"
echo -e "To activate the environment in a new terminal:"
echo -e "${BLUE}source $VENV_NAME/bin/activate${NC}"
echo -e "\nTo run the demo:"
echo -e "${BLUE}./run_demo.sh${NC}"
echo -e "\nTo deactivate the virtual environment when finished:"
echo -e "${BLUE}deactivate${NC}" 