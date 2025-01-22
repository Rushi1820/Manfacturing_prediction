from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pydantic import BaseModel
import os
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

router = APIRouter()

MODEL_PATH = "model.pkl"
DATA_PATH = "data/uploaded_data.csv"


class PredictionInput(BaseModel):
    Temperature: float
    Run_Time: int

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    
    df = pd.read_csv(file.file)
    
    # Validate required columns
    required_columns = {"Machine_ID", "Temperature", "Run_Time", "Downtime_Flag"}
    if not required_columns.issubset(df.columns):
        raise HTTPException(status_code=400, detail=f"CSV must contain columns: {required_columns}")

    # Save uploaded CSV file to the data directory
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    
    return {"message": "File uploaded successfully", "rows": len(df)}




@router.post("/train")
async def train_model():
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=400, detail="No data found. Please upload a CSV first.")
    
    # Load the dataset
    df = pd.read_csv(DATA_PATH)
    
    # Prepare the features and target variable
    X = df[['Temperature', 'Run_Time']]
    y = df['Downtime_Flag']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Hyperparameter tuning using GridSearchCV (Optional)
    param_grid = {
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10],
        'criterion': ['gini', 'entropy']
    }

    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best parameters and model
    model = grid_search.best_estimator_

    # Train the model using the best parameters
    model.fit(X_train, y_train)

    # Save trained model
    with open(MODEL_PATH, 'wb') as model_file:
        pickle.dump(model, model_file)

    # Evaluate model using StratifiedKFold Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    f1_scores = []

    for train_index, test_index in skf.split(X, y):
        X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
        y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_test_cv)

        accuracies.append(accuracy_score(y_test_cv, y_pred))
        f1_scores.append(f1_score(y_test_cv, y_pred))

    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    classification_rep = classification_report(y_train, y_train_pred, output_dict=True)

    return {
        "message": "Model trained successfully",
        "training_accuracy": round(train_accuracy, 2),
        "test_accuracy": round(test_accuracy, 2),
        "test_f1_score": round(test_f1, 2),
        "training_f1_score": round(train_f1, 2),
        "classification_report": classification_rep
    }


@router.post("/predict")
async def predict_downtime(data: PredictionInput):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")

    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)

    # Prepare input for prediction
    input_data = [[data.Temperature, data.Run_Time]]
    prediction_proba = model.predict_proba(input_data) 

    predicted_class = model.predict(input_data)[0]
    confidence_score = max(prediction_proba[0]) 

    result = "Yes" if predicted_class == 1 else "No"
    
    return {
        "Downtime": result,
        "Confidence": round(confidence_score, 2)
    }
