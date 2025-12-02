#!/bin/bash

# Multi-Objective Workflow Scheduler - Setup & Run Guide

set -e  # Exit on error

echo "========================================="
echo "Workflow Scheduler - Quick Start"
echo "========================================="

# Define venv directory
VENV_DIR="venv"

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in '$VENV_DIR'..."
    if ! python3 -m venv "$VENV_DIR"; then
        echo "ERROR: Failed to create virtual environment."
        echo "You may need to install the venv package. On Ubuntu/Debian:"
        echo "  sudo apt install python3.10-venv"
        exit 1
    fi
else
    echo "Virtual environment '$VENV_DIR' already exists."
fi

# Activate venv
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install dependencies
echo "Installing/Updating dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To run the scheduler, use the following command (ensure venv is activated):"
echo "  source $VENV_DIR/bin/activate"
echo "  python main.py --config config/config.yaml"
echo ""
echo "Or simply run this script again to setup and see instructions."
echo "========================================="

