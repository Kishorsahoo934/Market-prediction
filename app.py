from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd

# Initialize app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once (not inside route for efficiency)
model = pickle.load(open("simple_linear_regression.pkl", "rb"))

# Prediction endpoint
@app.get("/prediction/{budget}")
def predict(budget: int):
    # ✅ Match training feature name
    df = pd.DataFrame([[budget]], columns=['Marketing Budget (X) in Thousands'])

    prediction = model.predict(df)
    return {"prediction": int(prediction[1])}

