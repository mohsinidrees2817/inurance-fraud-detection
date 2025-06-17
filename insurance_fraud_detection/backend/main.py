from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import date, datetime
import joblib
import pandas as pd
import numpy as np
import logging
import json # Not used for file writing anymore, but good to keep if you need it elsewhere

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Insurance Fraud Detection API",
    description="Simplified API for detecting insurance fraud",
    version="1.0.0"
)

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"], # Ensure this matches your Streamlit port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic schemas
class ClaimInput(BaseModel):
    months_as_customer: int
    age: int
    policy_annual_premium: float
    umbrella_limit: int
    policy_bind_date: date # Pydantic will parse "YYYY-MM-DD" string into a date object
    policy_state: str
    insured_hobbies: str
    incident_type: str
    collision_type: str
    incident_severity: str
    incident_hour_of_the_day: int
    number_of_vehicles_involved: int
    witnesses: int
    total_claim_amount: float
    injury_claim: float
    property_claim: float
    vehicle_claim: float
    auto_model: str
    auto_year: int

class RiskFactor(BaseModel):
    factor: str
    description: str
    severity: Optional[str] = None

class PredictionOutput(BaseModel):
    prediction: int
    fraud_probability: float
    no_fraud_probability: float
    risk_factors: List[RiskFactor]
    safe_factors: List[Dict]
    comparison_metrics: Dict

# Load model and data
try:
    # IMPORTANT: Ensure these files exist in your backend folder when building the Docker image.
    model_artifacts = joblib.load("fraud_detection_model.pkl")
    sample_df = pd.read_csv("insurance_claims.csv")
    logger.info("Model and data loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or data: {str(e)}")
    # If loading fails, make sure the API doesn't crash but informs.
    # The /predict endpoint will then raise a 500 error, which is desired.
    model_artifacts = None
    sample_df = None

def get_feature_statistics():
    if sample_df is None:
        # Provide sensible defaults if sample_df failed to load
        logger.warning("Sample data not loaded, using default feature statistics.")
        return {'avg_claim': 30000, 'claim_75th': 50000, 'avg_age': 40, 'avg_premium': 1400}
    return {
        'avg_claim': sample_df['total_claim_amount'].mean(),
        'claim_75th': sample_df['total_claim_amount'].quantile(0.75),
        'avg_age': sample_df['age'].mean(),
        'avg_premium': sample_df['policy_annual_premium'].mean()
    }

def preprocess_input(input_data: dict) -> np.ndarray:
    processed_data = input_data.copy()
    current_year = datetime.now().year
    # Ensure 'auto_year' is an int before subtraction
    processed_data['vehicle_age'] = current_year - int(processed_data.get('auto_year', current_year))
    
    processed_data['number_of_vehicles_involved'] = max(1, processed_data.get('number_of_vehicles_involved', 1))
    # Handle division by zero if total_claim_amount is 0 or missing
    total_claim = processed_data.get('total_claim_amount', 0)
    num_vehicles = processed_data['number_of_vehicles_involved']
    processed_data['claim_per_vehicle'] = total_claim / num_vehicles if num_vehicles > 0 else 0
    
    processed_data['total_claim_ratio'] = 0 if total_claim == 0 else (
        (processed_data.get('injury_claim', 0) + processed_data.get('property_claim', 0) + processed_data.get('vehicle_claim', 0)) /
        total_claim
    )
    
    stats = get_feature_statistics()
    processed_data['high_value_claim'] = int(total_claim > stats['claim_75th'])
    incident_hour = processed_data.get('incident_hour_of_the_day', 0)
    processed_data['is_night_incident'] = int(incident_hour in range(0, 7) or 
                                            incident_hour in range(22, 24))
    
    label_encoders = model_artifacts['label_encoders']
    categorical_features = ['policy_bind_date', 'policy_state', 'insured_hobbies', 
                           'incident_type', 'collision_type', 'incident_severity', 'auto_model']
    
    # Special handling for policy_bind_date if it's a date object
    if isinstance(processed_data['policy_bind_date'], date):
        processed_data['policy_bind_date'] = processed_data['policy_bind_date'].isoformat() # Convert to string for encoder

    for col in categorical_features:
        try:
            # Use .get() with a default to avoid KeyError if a feature is missing
            # And ensure the input for transform is a list
            value_to_encode = processed_data.get(col, '') # Default to empty string if missing
            if col == 'policy_bind_date' and isinstance(value_to_encode, date):
                 value_to_encode = value_to_encode.isoformat()
            
            encoder = label_encoders.get(col)
            if encoder:
                # Handle unknown categories by using a default or mapping to 0
                try:
                    processed_data[col + '_encoded'] = encoder.transform([value_to_encode])[0]
                except ValueError:
                    logger.warning(f"Unknown category '{value_to_encode}' for feature '{col}'. Encoding as 0.")
                    processed_data[col + '_encoded'] = 0
            else:
                processed_data[col + '_encoded'] = 0 # No encoder found for this feature
        except Exception as e:
            logger.error(f"Error encoding {col}: {e}")
            processed_data[col + '_encoded'] = 0 # Fallback in case of encoding error
    
    selected_features = model_artifacts['selected_features']
    # Ensure all selected features are present, with a default of 0 if missing from input
    feature_vector = [processed_data.get(feature, 0) for feature in selected_features]
    feature_vector = [0 if np.isnan(v) or np.isinf(v) else v for v in feature_vector]
    
    return np.array(feature_vector).reshape(1, -1)

def analyze_risk_factors(input_data: dict) -> tuple[List[dict], List[dict]]:
    risk_factors, safe_factors = [], []
    stats = get_feature_statistics()
    
    # Check total_claim_amount vs average
    total_claim = input_data.get('total_claim_amount', 0)
    if total_claim > stats['avg_claim'] * 1.5:
        risk_factors.append({
            'factor': 'High Claim Amount',
            'description': f"Claim amount (${total_claim:,.0f}) exceeds average (${stats['avg_claim']:,.0f})",
            'severity': 'high'
        })
    elif total_claim < stats['avg_claim'] * 0.8:
        safe_factors.append({'factor': 'Reasonable Claim Amount', 'description': 'Claim within normal range'})
    
    # Check incident hour (night incident)
    incident_hour = input_data.get('incident_hour_of_the_day', 0)
    if incident_hour in range(0, 7) or incident_hour in range(22, 24):
        risk_factors.append({
            'factor': 'Night Incident',
            'description': f"Incident at {incident_hour}:00 (higher fraud risk)",
            'severity': 'medium'
        })
    else:
        safe_factors.append({'factor': 'Daytime Incident', 'description': 'Incident during daytime'})
    
    # Check vehicle age
    auto_year = input_data.get('auto_year', datetime.now().year)
    vehicle_age = datetime.now().year - auto_year
    if vehicle_age > 15:
        risk_factors.append({
            'factor': 'Old Vehicle',
            'description': f"Vehicle is {vehicle_age} years old",
            'severity': 'medium'
        })
    elif vehicle_age < 5:
        safe_factors.append({'factor': 'New Vehicle', 'description': f"Vehicle is {vehicle_age} years old"})
    
    # Check incident severity
    incident_severity = input_data.get('incident_severity', '')
    if incident_severity in ["Total Loss", "Major Damage"]:
        risk_factors.append({
            'factor': 'High Severity',
            'description': f"Severity: {incident_severity}",
            'severity': 'high'
        })
    elif incident_severity in ["Minor Damage", "Trivial Damage"]:
        safe_factors.append({'factor': 'Low Severity', 'description': f"Severity: {incident_severity}"})
    
    return risk_factors, safe_factors

@app.post("/predict", response_model=PredictionOutput)
async def predict(claim: ClaimInput):
    if model_artifacts is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded. Check server logs for details.")
    
    try:
        # Pydantic's .dict() converts the BaseModel to a Python dict
        # The policy_bind_date will already be a date object
        input_data = claim.dict()
        
        processed_input = preprocess_input(input_data)
        scaler = model_artifacts['scaler']
        scaled_input = scaler.transform(processed_input)
        
        model = model_artifacts['model']
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0]
        
        risk_factors, safe_factors = analyze_risk_factors(input_data)
        
        # Remove the file writing here, as the data is already returned in PredictionOutput
        # with open("risk_factors.json", "w") as f:
        #     json.dump({"risk_factors": risk_factors, "safe_factors": safe_factors}, f)
        
        stats = get_feature_statistics()
        comparison_metrics = {
            'claim_amount': {'user_value': input_data['total_claim_amount'], 'industry_avg': stats['avg_claim']},
            'customer_age': {'user_value': input_data['age'], 'industry_avg': stats['avg_age']},
            'annual_premium': {'user_value': input_data['policy_annual_premium'], 'industry_avg': stats['avg_premium']},
            'months_as_customer': {'user_value': input_data['months_as_customer'], 
                                  'industry_avg': sample_df['months_as_customer'].mean() if sample_df is not None else 200}
        }
        
        logger.info("Prediction completed successfully")
        return {
            "prediction": int(prediction),
            "fraud_probability": float(prediction_proba[1]),
            "no_fraud_probability": float(prediction_proba[0]),
            "risk_factors": risk_factors,
            "safe_factors": safe_factors,
            "comparison_metrics": comparison_metrics
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        # Raise HTTPException with a more informative message
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}. Check backend logs for more details.")

@app.get("/health")
async def health_check():
    # You can add checks here, e.g., if model_artifacts is loaded
    if model_artifacts is None:
        return {"status": "unhealthy", "message": "Model artifacts not loaded"}
    return {"status": "healthy"}