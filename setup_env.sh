#!/bin/bash
# Setup environment for Federated Learning project

export PYTHONPATH=$PYTHONPATH:D:\Federated Learning
export PYTHONPATH=$PYTHONPATH:D:\Federated Learning/src

echo "Environment variables set:"
echo "  PROJECT_ROOT=D:\Federated Learning"
echo "  PYTHONPATH=$PYTHONPATH"

echo ""
echo "Ready to run scripts!"
echo "Example: python src/training/local_trainer.py"
