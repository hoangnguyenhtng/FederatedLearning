@echo off
REM Setup environment for Federated Learning project

SET PYTHONPATH=%PYTHONPATH%;D:\Federated Learning
SET PYTHONPATH=%PYTHONPATH%;D:\Federated Learning\src

echo Environment variables set:
echo   PROJECT_ROOT=D:\Federated Learning
echo   PYTHONPATH=%PYTHONPATH%

echo.
echo Ready to run scripts!
echo Example: python src\training\local_trainer.py
