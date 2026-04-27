import joblib
import pandas as pd
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "predict_freight_model.pkl"

def load_model(model_path: str = MODEL_PATH):
    """Load trained freight cost prediction model"""
    with open(model_path, 'rb') as f:
        data = joblib.load(f)
    return data["model"], data["features"]

def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices,
    Parameters:
    ----------------
    input_data: dict
    Returns
    ----------------
    pd.DataFrame with predicted freight cost
    """
    model, features = load_model()
    input_df = pd.DataFrame(input_data)
    input_df = input_df[features]
    input_df['Predicted_Freight'] = model.predict(input_df).round()
    return input_df

if __name__ == "__main__":

    #Example inference run(local testing)
    sample_data = {
        "Dollar_transformed": [10,11],
    }
    prediction = predict_freight_cost(sample_data)
    print(prediction)