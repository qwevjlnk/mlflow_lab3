from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from mlflow.models import infer_signature

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


def make_ohe():
    """Создаёт OneHotEncoder, совместимый с разными версиями scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def prepare_features(df: pd.DataFrame):
    data = df.copy()

    # Берём бренд из названия машины: это полезнее, чем кодировать весь name,
    # потому что у name очень много уникальных значений.
    data["brand"] = data["name"].astype(str).str.split().str[0].fillna("Unknown")

    # Возраст машины
    current_year = pd.Timestamp.today().year
    data["car_age"] = current_year - data["year"]

    feature_cols = [
        "car_age",
        "km_driven",
        "fuel",
        "seller_type",
        "transmission",
        "owner",
        "brand",
    ]

    X = data[feature_cols].copy()
    y = data["selling_price"].copy()
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = ["car_age", "km_driven"]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_ohe()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )
    return preprocessor


def build_models() -> dict:
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_one_model(model_name, estimator, X_train, X_test, y_train, y_test, preprocessor, base_dir: Path):
    regressor = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    # Логарифмируем цену, чтобы модель лучше училась на "перекошенных" ценах
    wrapped = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tag("dataset", "CAR DETAILS FROM CAR DEKHO")
        mlflow.set_tag("target", "selling_price")
        mlflow.set_tag(
            "features",
            "car_age, km_driven, fuel, seller_type, transmission, owner, brand",
        )
        mlflow.set_tag("model_name", model_name)

        # Логируем только простые параметры модели
        for key, value in estimator.get_params().items():
            if isinstance(value, (str, int, float, bool, type(None))):
                mlflow.log_param(key, value)

        mlflow.log_param("target_transform", "log1p/expm1")

        wrapped.fit(X_train, y_train)
        y_pred = wrapped.predict(X_test)

        metrics = evaluate(y_test, y_pred)
        mlflow.log_metrics(metrics)

        # Пример входа + signature нужны, чтобы удобно запускать serve
        input_example = X_train.head(1)
        signature = infer_signature(input_example, wrapped.predict(input_example))
        mlflow.sklearn.log_model(
            sk_model=wrapped,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        # Несколько предсказаний сохраняем как файл-артефакт
        sample = X_test.head(10).copy()
        sample["actual_price"] = y_test.head(10).values
        sample["predicted_price"] = wrapped.predict(sample)

        sample_path = base_dir / f"{model_name}_sample_predictions.csv"
        sample.to_csv(sample_path, index=False)
        mlflow.log_artifact(str(sample_path), artifact_path="samples")

        return run.info.run_id, metrics


def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "CAR DETAILS FROM CAR DEKHO.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Не найден файл: {csv_path}")

    # Лучше хранить всё в папке проекта
    tracking_uri = base_dir.joinpath("mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("car_price_experiment")

    df = load_data(csv_path)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(X_train)
    models = build_models()

    results = []

    for model_name, estimator in models.items():
        print(f"\nОбучаю модель: {model_name}")
        run_id, metrics = train_one_model(
            model_name=model_name,
            estimator=estimator,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            preprocessor=preprocessor,
            base_dir=base_dir,
        )
        results.append(
            {
                "model": model_name,
                "run_id": run_id,
                **metrics,
            }
        )
        print(f"run_id: {run_id}")
        print(f"RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R2: {metrics['r2']:.4f}")

    results_df = pd.DataFrame(results).sort_values(
        by=["rmse", "mae"],
        ascending=True,
    )

    results_path = base_dir / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    best_row = results_df.iloc[0]
    best_info_path = base_dir / "best_model_info.txt"
    best_info_path.write_text(
        "\n".join(
            [
                f"best_model={best_row['model']}",
                f"run_id={best_row['run_id']}",
                f"rmse={best_row['rmse']}",
                f"mae={best_row['mae']}",
                f"r2={best_row['r2']}",
            ]
        ),
        encoding="utf-8",
    )

    print("\nСравнение моделей:")
    print(results_df.to_string(index=False))

    print(f"\nЛучшая модель: {best_row['model']}")
    print(f"run_id лучшей модели: {best_row['run_id']}")
    print(f"Таблица сравнения сохранена в: {results_path}")
    print(f"Информация о лучшей модели сохранена в: {best_info_path}")

    print("\nКоманда для запуска сервиса:")
    print(f"mlflow models serve -m runs:/{best_row['run_id']}/model --host 127.0.0.1 --port 5000 --env-manager local")


if __name__ == "__main__":
    main()
