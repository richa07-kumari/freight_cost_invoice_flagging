from data_preprocessing import load_invoice_data, split_data, scale_features, apply_labels
from modelling_evaluation import train_random_forest, evaluate_classifier
import joblib
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]

FEATURES =['invoice_quantity','invoice_dollars','Freight','total_item_quantity','total_item_dollars','avg_receiving_delay']

TARGET = 'flag_invoice'

def main():
    model_dir = BASE_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    
    #Load data
    df = load_invoice_data()
    df = apply_labels(df)
    if df is None or df.empty:
        raise ValueError("Data loading failed")

    #Prepare data
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, model_dir / "scaler.pkl")

    #Train and evaluate model
    model = train_random_forest(X_train_scaled, y_train)

    evaluate_classifier(model, X_test_scaled, y_test, "Random Forest Classifier")

    #Save model
    model_path = model_dir / "predict_flag_invoice.pkl"
    joblib.dump({
        "model": model,
        "features": FEATURES
    }, model_path)
    print("Random Forest model saved.")
    print(f"Model path : {model_path}")

if __name__ == "__main__":
    main()