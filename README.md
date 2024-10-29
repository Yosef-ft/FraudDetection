# Fraud Detection System
## Overview
This project aims to develop robust and accurate fraud detection models. he models are designed to enhance the detection of fraudulent transactions for e-commerce and bank credit transactions. The solution leverages machine learning models, geolocation analysis, and transaction pattern recognition to improve fraud detection accuracy.

## Project Goals
1. Data Analysis and Preprocessing
2. Tracking file changes using DVC
3. Model Building and Training
4. Model Explainability analysis
5. Model Deployment and API Development
6. Build a Dashboard with Flask and Dash

## Getting Started
### Prerequisites
Make sure you have the following installed:
  * Python 3.x
  * Pip (Python package manager)

### Installation
Clone the repository:
```
git clone https://github.com/Yosef-ft/FraudDetection.git
cd FraudDetection
```
Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
Install the required packages:
```
pip install -r requirements.txt
```

### Viewing the Dashboard
To view the dashboard, follow these steps:

1. Run Notebooks: Run all the Jupyter notebooks in the repository.
2. Create a Report Folder: In the Flask_Dash directory, create a folder named report.
3. Download Experiment Plots from MLflow: Inside MLflow, download the experiment plots. Save them to the Flask_Dash/report directory.

Run Docker Compose:
Use the following command to build and start the Docker containers:
```
docker compose up --build
```
