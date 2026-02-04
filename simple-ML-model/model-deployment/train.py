import mlflow
import mlflow.sklearn
import joblib
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "logistic_regression": LogisticRegression(max_iter=200),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

mlflow.set_experiment("Simple-ML-Comparison")

best_model = None
best_accuracy = 0.0

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, name)

        print(f"{name} accuracy: {acc}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model

# Save best model
joblib.dump(best_model, "model.pkl")
print("Best model saved as model.pkl")
