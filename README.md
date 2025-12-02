Turbine Thermal MLOps

End-to-End Machine Learning + MLOps System for Turbine Blade Temperature & Risk Prediction

This repository contains a complete machine learning + MLOps pipeline for predicting gas turbine blade surface temperature and classifying operational risk (safe/warning/critical).
It includes:

Data preprocessing & feature engineering

Model training & evaluation

MLflow experiment tracking

FastAPI inference server

Dockerized deployment

Prefect workflows for retraining & batch prediction

This project demonstrates a real-world production ML workflow.

Tech Stack

ML & Data: Python, Pandas, NumPy, Scikit-Learn
Experiment Tracking: MLflow
Serving: FastAPI + Uvicorn
Containerization: Docker
Orchestration: Prefect
Environment: Conda / pip

 Model Training Workflow
Run preprocessing + training:
python src/data/features.py
python src/models/train.py


This:

Engineers features

Splits dataset

Scales inputs

Trains temperature regression model

Trains risk classification model

Logs everything to MLflow

Saves models into models/

 FastAPI Model Server

Start the prediction server:

uvicorn src.api.app:app --reload


API endpoints:

Path	Description
/	Health check
/predict	Predict temperature & risk
/docs	Interactive Swagger UI
Sample Input (JSON)
{
  "rpm": 9000,
  "load_pct": 0.85,
  "t_inlet_K": 1350,
  "p_inlet_bar": 18,
  "t_coolant_K": 420,
  "m_dot_coolant_kg_s": 0.18,
  "fuel_h2_frac": 0.5,
  "k_material_W_mK": 25,
  "blade_thickness_m": 0.004,
  "chord_length_m": 0.11
}

 Docker Deployment

Build the image:

docker build -f docker/Dockerfile.api -t turbine-api .


Run the container:

docker run -p 8000:8000 turbine-api


API will be available at:

 http://127.0.0.1:8000

 http://127.0.0.1:8000/docs

 Prefect Orchestration
 Automated Model Retraining

Runs preprocessing + training:

python orchestration/training_flow.py

 Batch Predictions on Daily Logs

Runs inference on new turbine sensor logs:

python orchestration/batch_prediction_flow.py


Outputs are saved to:

data/predictions/

 Architecture (Mermaid Diagram)
flowchart TD

A[Raw Data] --> B[Feature Engineering]

B --> C[Train Regression Model]

B --> D[Train Classification Model]


C --> E[MLflow Tracking]

D --> E



E --> F[Saved Models]


F --> G[FastAPI Inference Server]

G --> H[Docker Deployment]


A2[Daily Logs] --> I[Batch Prediction Flow]

I --> G


subgraph Prefect Flows
    B
    C
    D
    I
end

 Highlights

Realistic synthetic turbine operating dataset

Regression & classification models

Automated MLOps pipelines

Fully containerized API

Clean, production-style project layout

Ready for cloud deployment (Render/Railway/Cloud Run)

 Author

Om Ray (1225zisu-lab)


Mechanical Engineering + ML Engineer

Turbine Heat Transfer & Hydrogen Fuel Research | MLOps Enthusiast
