from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion='gini',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model

def evaluate_classifier(model, X_test, y_test, model_name):
    pred = model.predict(X_test)
    
    print(f"\n{model_name} performance:")
    print(f"Accuracy : {model.score(X_test, y_test)*100:.2f}%")
    print(f"F1 Score : {f1_score(y_test, pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, pred))

    return