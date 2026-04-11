import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_cleaning import load_data, clean_data
from feature_engineering import create_features, create_target


MODEL_PATH = "risk_model.pkl"


def build_training_pipeline():
    numeric_features = [
        "scheduled_time",
        "actual_time",
        "delay",
        "pickup_lat",
        "pickup_lon",
        "drop_lat",
        "drop_lon",
        "distance_km",
        "traffic_severity",
        "priority_score",
        "is_late_start",
    ]

    categorical_features = [
        "priority",
        "traffic_level",
        "status",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline, numeric_features + categorical_features


def main():
    file_path = "data/orders_with_locations.csv"

    df = load_data(file_path)
    df = clean_data(df)
    df = create_features(df)
    df = create_target(df, fail_delay_threshold=15)

    pipeline, feature_columns = build_training_pipeline()

    X = df[feature_columns]
    y = df["will_fail"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    print("\n===== MODEL EVALUATION =====")
    print(classification_report(y_test, preds))
    print("ROC AUC:", round(roc_auc_score(y_test, probs), 4))

    joblib.dump(
        {
            "model": pipeline,
            "feature_columns": feature_columns,
        },
        MODEL_PATH,
    )

    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()