from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, validator
import pandas as pd
import pickle

app = FastAPI(title="Titanic Survival Prediction API")

# Set up template rendering
templates = Jinja2Templates(directory="templates")

# Load model and scaler once at startup
async def load_model_and_scaler():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("standard_scalar.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# Data schema with validations
class DataModel(BaseModel):
    PClass: int = Field(..., ge=1, le=3, description="Passenger class must be 1, 2, or 3")
    Age: float = Field(..., gt=0, lt=120, description="Age must be a positive number less than 120")
    Sex: int = Field(..., ge=0, le=1, description="Sex must be 0 (female) or 1 (male)")

    @validator("PClass")
    def validate_pclass(cls, value):
        if value not in [1, 2, 3]:
            raise ValueError("PClass must be 1, 2, or 3")
        return value

    @validator("Sex")
    def validate_sex(cls, value):
        if value not in [0, 1]:
            raise ValueError("Sex must be 1 (female) or 0 (male)")
        return value

# Homepage route (HTML form)
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction route
@app.post("/predict")
async def predict(data: DataModel):
    """
    Predict whether a Titanic passenger survived or not.
    """
    try:
        
        input_df = pd.DataFrame([data.dict()])
        
        model, scaler = await load_model_and_scaler()

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        return {
            "prediction": int(prediction[0]),
            "message": "Passenger survived" if prediction[0] == 1 else "Passenger did not survive"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
