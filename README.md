# Machine Downtime Prediction API
This FastAPI-based application allows users to predict machine downtime using a supervised machine learning model. The system provides endpoints to upload manufacturing data, train a machine learning model, and make predictions based on temperature and runtime values.

# Features
Upload a CSV dataset containing machine-related data.
Train a machine learning model (Decision Tree Classifier) to predict machine downtime.
Make predictions with confidence scores based on new input data.

# Generate Synthetic Data
Use the datagenerator.py script to generate synthetic data with key columns like Machine_ID, Temperature, Run_Time, Downtime_Flag. 

# API Endpoints
# 1. Upload Dataset
Endpoint: POST /upload, "http://localhost:8000/upload"
Description: Upload a CSV file containing manufacturing data.
# Request:
A CSV file with columns: Machine_ID, Temperature, Run_Time, Downtime_Flag.
# Response:

{
  "message": "File uploaded successfully",
  "rows": 1000
}

# 2. Train Model
Endpoint: POST /train, "http://localhost:8000/train"
Description: Train a Decision Tree model on the uploaded dataset and return model performance metrics.
# Request: No request body needed.
# Response:
{
  "message": "Model trained successfully",
  "training_accuracy": 0.51,
  "test_accuracy": 0.5,
  "test_f1_score": 0.33,
  "training_f1_score": 0.33,
  "classification_report": {
    "0": {
      "precision": 0.5053583444110744,
      "recall": 0.7681734271316669,
      "f1-score": 0.6096476900385075,
      "support": 40086
    },
    "1": {
      "precision": 0.5126134158493733,
      "recall": 0.24487648444154933,
      "f1-score": 0.33142876519557146,
      "support": 39914
    },
    "accuracy": 0.5070875,
    "macro avg": {
      "precision": 0.5089858801302238,
      "recall": 0.5065249557866082,
      "f1-score": 0.47053822761703945,
      "support": 80000
    },
    "weighted avg": {
      "precision": 0.5089780809284277,
      "recall": 0.5070875,
      "f1-score": 0.4708373129612456,
      "support": 80000
    }
  }
}
# 3. Predict Downtime
Endpoint: POST /predict, "http://localhost:8000/predict"
Description: Predict machine downtime (Yes/No) based on input values for Temperature and Run_Time.
# Request:
 {
  "Temperature": 98.10, 
  "Run_Time": 420
  }
# Response:
{
  "Downtime": "Yes",
  "Confidence": 0.85
}

# Setup Instructions
# Install Dependencies
Install the required Python packages:
pip install **pip install -r .\requirements.txt**

# Run the API
Start the FastAPI server:
**uvicorn main:app --reload**
The API will be available at **http://localhost:8000**

# Access Documentation
Swagger UI: http://localhost:8000/docs

# Technologies Used
FastAPI: For building the REST API.
Uvicorn: ASGI server to run the FastAPI application.
Pandas: For data manipulation.
Scikit-Learn: For building and training the machine learning model.


email: rushivardhan18@gmail.com
phone no: 9603366515
