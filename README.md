
# Heart Disease Prediction App

A machine learning web application built using Python,
FastAPI, Scikit-learn, HTML, CSS, and JavaScript.

## Features
- Heart disease prediction form
- Logistic Regression model
- Model probability estimate
- FastAPI prediction endpoint
- Responsive web interface
- Vercel deployment configuration

## Project Structure
- api/: FastAPI application and prediction logic
- data/: Heart disease dataset
- model/: Saved model and preprocessing files
- public/: Frontend files
- train_model.py: Model training script

## Run Locally

1. Install Python 3.10 or a compatible Python version.
2. Create and activate a virtual environment.
3. Install requirements:

   pip install -r requirements.txt

4. Put your compatible dataset at data/heart.csv.
5. Train the model:

   python train_model.py

6. Start the application from the project root:

   uvicorn api.index:app --reload

7. Open http://127.0.0.1:8000

## Deployment

Push the project to GitHub and import the repository into Vercel.
Keep vercel.json in the root directory and check the deployment logs.

## Important

This is an educational project, not a medical diagnostic tool.
Predictions depend on the dataset and model quality.
