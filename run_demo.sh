#!/bin/bash
# Ray + Iceberg + OpenLineage Demo

# Colors for pretty output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BLUE}${BOLD}Ray + Iceberg + OpenLineage Demo${NC}"
echo "=========================================================="

# Check if running in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: No active Python virtual environment detected.${NC}"
    echo -e "${YELLOW}It's recommended to run this demo in a virtual environment:${NC}"
    echo -e "${BLUE}python3 -m venv ray_lineage_venv${NC}"
    echo -e "${BLUE}source ray_lineage_venv/bin/activate${NC}"
    echo -e "${BLUE}pip install -r requirements.txt${NC}"
    echo ""
    read -p "Continue anyway? (y/n): " continue_without_venv
    if [[ $continue_without_venv != "y" && $continue_without_venv != "Y" ]]; then
        echo "Exiting. Please set up a virtual environment before running the demo."
        exit 1
    fi
else
    echo -e "${GREEN}✓ Using virtual environment: $(basename $VIRTUAL_ENV)${NC}"
fi

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is required but not installed.${NC}"
    exit 1
fi

if ! docker compose version &> /dev/null && ! docker-compose --version &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is required but not installed.${NC}"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
    exit 1
fi

# Check for required packages
echo -e "${YELLOW}Checking required packages...${NC}"
python3 -c "import ray, pandas, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Installing required packages...${NC}"
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install required packages.${NC}"
        exit 1
    fi
fi

# Check specifically for tensorflow
python3 -c "import tensorflow" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}TensorFlow not found. Installing...${NC}"
    pip install "tensorflow>=2.10.0"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install TensorFlow. Demo may not work correctly.${NC}"
        echo -e "${YELLOW}If you encounter errors, try installing TensorFlow manually:${NC}"
        echo -e "${BLUE}pip install tensorflow>=2.10.0${NC}"
    fi
fi

echo -e "${GREEN}✓ Required packages are installed${NC}"

# Create storage directories
mkdir -p storage/datasets storage/models

# Start Marquez components
echo -e "${YELLOW}Starting Marquez components with Docker Compose...${NC}"

# Check which docker-compose command works
if command -v docker compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
else
    DOCKER_COMPOSE_CMD="docker-compose"
fi

# Check if Marquez is already running
if $DOCKER_COMPOSE_CMD ps | grep -q "marquez"; then
    echo -e "${GREEN}✓ Marquez components are already running${NC}"
else
    # Start only Postgres, Marquez and Web components
    $DOCKER_COMPOSE_CMD up -d postgres marquez web
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to start Marquez components.${NC}"
        echo -e "${RED}Check docker-compose.yml and Docker logs for more information.${NC}"
        exit 1
    fi
    
    # Wait for Marquez to start
    echo -e "${YELLOW}Waiting for Marquez to start (this might take a minute)...${NC}"
    
    # Try up to 30 times (waiting 2 seconds between attempts)
    for i in {1..30}; do
        sleep 2
        # Check if Marquez API is responding
        curl -s http://localhost:5002/api/v1/namespaces > /dev/null
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Marquez is now running${NC}"
            break
        fi
        echo -n "."
        if [ $i -eq 30 ]; then
            echo ""
            echo -e "${RED}Timed out waiting for Marquez to start.${NC}"
            echo -e "${YELLOW}Continuing anyway, but lineage tracking might not work.${NC}"
        fi
    done
fi

echo -e "${GREEN}✓ Marquez UI is available at: ${BLUE}http://localhost:3000${NC}"

# Run the demo script
echo -e "${YELLOW}Starting demo...${NC}"
python3 ray_iceberg_lineage.py

# Script will handle its own cleanup on exit
echo -e "${BLUE}${BOLD}When done, you can stop Marquez with:${NC}"
echo -e "${DOCKER_COMPOSE_CMD} down" 