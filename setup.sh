#!/bin/bash

# Entity Resolution Pipeline Setup Script
# This script sets up the environment for the entity resolution pipeline

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# Print header
echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}Entity Resolution Pipeline Setup${NC}"
echo -e "${BLUE}=================================${NC}"

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1)
if [[ $python_version == *"Python 3."* ]]; then
    echo -e "${GREEN}Python detected: $python_version${NC}"
else
    echo -e "${RED}Error: Python 3.9+ is required. Found: $python_version${NC}"
    echo -e "${YELLOW}Please install Python 3.9+ and try again${NC}"
    exit 1
fi

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate || (echo -e "${RED}Failed to activate virtual environment${NC}" && exit 1)
echo -e "${GREEN}Virtual environment activated${NC}"

# Install requirements
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt || (echo -e "${RED}Failed to install dependencies${NC}" && exit 1)
echo -e "${GREEN}Dependencies installed successfully${NC}"

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p data/input
mkdir -p data/ground_truth
mkdir -p checkpoints
mkdir -p output
mkdir -p tmp
echo -e "${GREEN}Directories created${NC}"

# Check Docker and Docker Compose
echo -e "${YELLOW}Checking Docker and Docker Compose...${NC}"
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}Docker and Docker Compose detected${NC}"
    
    # Check if Weaviate is already running
    if docker ps | grep -q weaviate; then
        echo -e "${YELLOW}Weaviate is already running${NC}"
    else
        echo -e "${YELLOW}Starting Weaviate and monitoring services...${NC}"
        docker-compose up -d || (echo -e "${RED}Failed to start Weaviate${NC}" && exit 1)
        echo -e "${GREEN}Weaviate and monitoring services started${NC}"
        
        # Wait for Weaviate to be ready
        echo -e "${YELLOW}Waiting for Weaviate to be ready...${NC}"
        attempt=0
        max_attempts=30
        while [ $attempt -lt $max_attempts ]; do
            if curl -s http://localhost:8080/v1/.well-known/ready | grep -q "true"; then
                echo -e "${GREEN}Weaviate is ready${NC}"
                break
            fi
            attempt=$((attempt+1))
            echo -e "${YELLOW}Waiting for Weaviate (attempt $attempt/$max_attempts)...${NC}"
            sleep 5
        done
        
        if [ $attempt -eq $max_attempts ]; then
            echo -e "${RED}Weaviate did not become ready in time${NC}"
            echo -e "${YELLOW}Check Docker logs: docker logs weaviate${NC}"
        fi
    fi
else
    echo -e "${RED}Docker and/or Docker Compose not found${NC}"
    echo -e "${YELLOW}Please install Docker and Docker Compose and try again${NC}"
    echo -e "${YELLOW}Skip this check if you're using a remote Weaviate instance${NC}"
fi

# Check OpenAI API key
echo -e "${YELLOW}Checking OpenAI API key...${NC}"
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}OPENAI_API_KEY environment variable not set${NC}"
    echo -e "${YELLOW}Please set your OpenAI API key:${NC}"
    echo -e "${YELLOW}export OPENAI_API_KEY=your_api_key${NC}"
else
    echo -e "${GREEN}OpenAI API key detected${NC}"
fi

# Check available system resources
echo -e "${YELLOW}Checking system resources...${NC}"
total_memory=$(grep MemTotal /proc/meminfo | awk '{print $2}')
total_memory_gb=$((total_memory / 1024 / 1024))
cpu_cores=$(grep -c ^processor /proc/cpuinfo)

echo -e "${GREEN}Available memory: ${total_memory_gb}GB${NC}"
echo -e "${GREEN}Available CPU cores: ${cpu_cores}${NC}"

if [ $total_memory_gb -lt 16 ]; then
    echo -e "${YELLOW}Warning: Less than 16GB of RAM detected. Development mode only.${NC}"
elif [ $total_memory_gb -lt 32 ]; then
    echo -e "${YELLOW}Warning: Less than 32GB of RAM detected. Limited processing capacity.${NC}"
fi

if [ $cpu_cores -lt 4 ]; then
    echo -e "${YELLOW}Warning: Less than 4 CPU cores detected. Processing will be slow.${NC}"
fi

# Setup complete
echo -e "${BLUE}=================================${NC}"
echo -e "${GREEN}Setup completed successfully${NC}"
echo -e "${BLUE}=================================${NC}"
echo -e "${YELLOW}To run the pipeline:${NC}"
echo -e "${YELLOW}1. Make sure your virtual environment is activated:${NC}"
echo -e "   ${BLUE}source venv/bin/activate${NC}"
echo -e "${YELLOW}2. Run the pipeline in development mode:${NC}"
echo -e "   ${BLUE}python main.py --mode dev${NC}"
echo -e "${YELLOW}3. Or run specific components:${NC}"
echo -e "   ${BLUE}python main.py --component preprocess${NC}"
echo -e "${BLUE}=================================${NC}"
