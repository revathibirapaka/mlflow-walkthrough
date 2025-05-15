# MLflow Walkthrough

This repository demonstrates a basic walkthrough of using [MLflow](https://mlflow.org/) for experiment tracking with a sample machine learning model.

## Overview

The goal of this walkthrough is to:

- Train a sample machine learning model.
- Use MLflow to log parameters, metrics, and the model artifact.
- Launch the MLflow tracking UI to view the logged data.

## Files

- `ml_flow.py`: Python script that trains a `RandomForestRegressor` model using the diabetes dataset, logs parameters and metrics using MLflow, and stores the trained model.

## ▶How to Run

### 1. Create and activate conda environment

```bash
conda create -n mlflow_env python=3.10 -y
conda activate mlflow_env

### **2. Install dependencies**

```bash
pip install mlflow scikit-learn pandas

### **3. Run the script**

```bash

python ml_flow.py

### **4. Launch MLflow UI**

```bash

mlflow ui

Then open the browser and go to:

http://localhost:5000
