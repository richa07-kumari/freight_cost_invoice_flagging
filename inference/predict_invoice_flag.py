import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "predict_flag_invoice.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

def load_model(model_path: str = MODEL_PATH):
    with open(model_path, "rb") as f:
        data = joblib.load(f)

    model = data["model"]
    features = data["features"]

    scaler = joblib.load(SCALER_PATH)

    return model, features, scaler

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.
    Parameters
    ----------
    input_data: dict
    Returns
    ----------
    pd.DataFrame with predicted flag
    """
    model, features, scaler = load_model()
    input_df = pd.DataFrame(input_data)
    input_df = input_df[features]
    input_scaled = scaler.transform(input_df)
    input_df['Predicted_Flag'] = model.predict(input_scaled).round()
    return input_df

if __name__ == "__main__":
    sample_data = {
    "invoice_quantity": [1000, 500, 200, 1500, 300, 800],
    "invoice_dollars":  [10000, 5200, 2100, 16000, 4000, 8500],
    "Freight":          [55, 30, 15, 95, 40, 60],
    "total_item_quantity": [1000, 500, 200, 1400, 300, 800],
    "total_item_dollars":  [10000, 5000, 2000, 15000, 3500, 8500],
    "avg_receiving_delay": [5, 7, 3, 12, 15, 6]
    }    
    prediction = predict_invoice_flag(sample_data)
    print(prediction)