# src/models/train.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import joblib
from pathlib import Path

def load_processed_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_reg_train = pd.read_csv("data/processed/y_reg_train.csv")
    y_reg_test = pd.read_csv("data/processed/y_reg_test.csv")
    y_class_train = pd.read_csv("data/processed/y_class_train.csv")
    y_class_test = pd.read_csv("data/processed/y_class_test.csv")

    return X_train, X_test, y_reg_train.values.ravel(), y_reg_test.values.ravel(), y_class_train.values.ravel(), y_class_test.values.ravel()

def train_regression(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    return model, mae, rmse

def train_classification(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return model, acc

def main():
    print("Loading processed data...")
    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = load_processed_data()

    mlflow.set_experiment("turbine_thermal_model")

    with mlflow.start_run():
        print("Training regression model...")
        reg_model, mae, rmse = train_regression(X_train, X_test, y_reg_train, y_reg_test)

        print("Training classification model...")
        class_model, acc = train_classification(X_train, X_test, y_class_train, y_class_test)

        # log params
        mlflow.log_param("reg_n_estimators", 200)
        mlflow.log_param("reg_max_depth", 12)

        mlflow.log_param("class_n_estimators", 300)
        mlflow.log_param("class_max_depth", 10)

        # log metrics
        mlflow.log_metric("reg_mae", mae)
        mlflow.log_metric("reg_rmse", rmse)
        mlflow.log_metric("classification_accuracy", acc)

        print("\nModel Performance:")
        print(f"Regression MAE: {mae}")
        print(f"Regression RMSE: {rmse}")
        print(f"Classification Accuracy: {acc}")

        Path("models/").mkdir(exist_ok=True)

        joblib.dump(reg_model, "models/regression_model.pkl")
        joblib.dump(class_model, "models/classification_model.pkl")

        mlflow.sklearn.log_model(reg_model, artifact_path="regression_model")
        mlflow.sklearn.log_model(class_model, artifact_path="classification_model")

        print("\n✔ Models saved in: models/")
        print("✔ Models logged to MLflow.")

if __name__ == "__main__":
    main()
