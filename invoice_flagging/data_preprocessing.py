import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_invoice_data():
    conn = sqlite3.connect("../data/inventory.db")

    query = """
    WITH purchase_agg AS (
    SELECT 
        p.PONumber,
        COUNT(DISTINCT Brand) as total_brands,
        SUM(Quantity) as total_item_quantity,
        SUM(Dollars) as total_item_dollars,
        AVG(JULIANDAY(ReceivingDate) - JULIANDAY(PODate)) as avg_receiving_delay
    FROM purchases p
    GROUP BY p.PONumber
    )
    SELECT
        vi.PONumber,
        vi.Quantity as invoice_quantity,
        vi.Dollars as invoice_dollars,
        vi.Freight,
        (JULIANDAY(vi.InvoiceDate) - JULIANDAY(vi.PODate)) as days_po_to_invoice,
        (JULIANDAY(vi.PayDate) - JULIANDAY(vi.InvoiceDate)) as days_to_pay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay
    
    FROM vendor_invoice vi
    LEFT JOIN purchase_agg pa USING (PONumber)
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def apply_labels(df):
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty")
    df = df.copy()
    
    condition1 = abs(df["invoice_dollars"] - df["total_item_dollars"]) > 0.05
    condition2 = df["avg_receiving_delay"] > 10
    
    df["flag_invoice"] = (condition1 | condition2).astype(int)

    return df

def split_data(df, features, target):
    X = df[features]
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42)

def scale_features(X_train, X_test, scaler_path):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, scaler_path)
    return X_train_scaled, X_test_scaled
    