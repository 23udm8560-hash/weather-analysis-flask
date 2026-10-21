import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(BASE_DIR, "data", "weather_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILE = os.path.join(MODEL_DIR, "weather_model.joblib")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")


# Features the model will use.
NUMERIC_FEATURES = [
    "previous_temperature",
    "previous_humidity",
    "previous_rainfall",
    "previous_wind_speed",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
]

CATEGORICAL_FEATURES = ["city"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "temperature"


def load_and_prepare_data():
    """Load the CSV and create next-hour prediction features."""

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Weather dataset not found: {DATA_FILE}\n"
            "Run your Flask app and collect weather data first."
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = [
        "time",
        "city",
        "temperature",
        "humidity",
        "rainfall",
        "wind_speed",
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in CSV: {sorted(missing)}"
        )

    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    numeric_columns = [
        "temperature",
        "humidity",
        "rainfall",
        "wind_speed",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df.dropna(
        subset=["time", "city", "temperature"]
    )

    df["city"] = df["city"].astype(str).str.strip()

    df = df[df["city"] != ""]

    df = df.drop_duplicates(
        subset=["city", "time"]
    )

    df = df.sort_values(["city", "time"]).copy()

    # Previous hour's measurements become input features.
    grouped = df.groupby("city", sort=False)

    df["previous_temperature"] = grouped["temperature"].shift(1)
    df["previous_humidity"] = grouped["humidity"].shift(1)
    df["previous_rainfall"] = grouped["rainfall"].shift(1)
    df["previous_wind_speed"] = grouped["wind_speed"].shift(1)

    # Calendar features for the hour being predicted.
    hour = df["time"].dt.hour
    day = df["time"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df["day_sin"] = np.sin(2 * np.pi * day / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * day / 365.25)

    # Drop the first record for each city because it has no
    # previous weather observation.
    df = df.dropna(
        subset=[
            "previous_temperature",
            "previous_humidity",
            "previous_rainfall",
            "previous_wind_speed",
        ]
    )

    df = df.sort_values("time").reset_index(drop=True)

    return df


def train_model():
    """Train, evaluate and save the temperature prediction model."""

    print("Loading weather dataset...")

    df = load_and_prepare_data()

    print(f"Usable records: {len(df)}")
    print(f"Cities: {df['city'].nunique()}")

    if len(df) < 50:
        raise ValueError(
            "Not enough usable records for this demonstration. "
            "Collect more weather data before training."
        )

    # Keep the final 20% of observations for testing.
    split_index = int(len(df) * 0.80)

    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Unable to create a valid train/test split.")

    X = df[FEATURES]
    y = df[TARGET]

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Process numeric and categorical features separately.
    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    print("Training Random Forest model...")

    model.fit(X_train, y_train)

    print("Evaluating model...")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    # A simple baseline: predict the previous observed temperature.
    baseline_predictions = X_test["previous_temperature"]

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions,
    )

    metrics = {
        "model": "Random Forest Regression",
        "target": "Next available hourly temperature (C)",
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "cities": sorted(df["city"].unique().tolist()),
        "mae": round(float(mae), 3),
        "rmse": round(float(rmse), 3),
        "r2": round(float(r2), 3),
        "previous_temperature_baseline_mae": round(
            float(baseline_mae), 3
        ),
    }

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        MODEL_FILE,
    )

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nModel training completed!")
    print(f"Model saved to: {MODEL_FILE}")

    print("\nEvaluation results:")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    if mae >= baseline_mae:
        print(
            "\nNote: The model did not outperform the "
            "previous-temperature baseline on this test set."
        )

    return model, metrics


if __name__ == "__main__":
    train_model()
