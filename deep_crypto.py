# ==========================================================
# CryptoCast - TensorFlow Full Project
# Multi-Horizon Bitcoin Forecasting
# Models:
# 1D-CNN | RNN | LSTM | Transformer
# Metrics:
# MAE | RMSE | MAPE | R2
# ==========================================================

# ==========================================================
# IMPORTS
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

import tensorflow as tf

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Conv1D,
    GlobalAveragePooling1D,
    SimpleRNN,
    LSTM,
    LayerNormalization,
    MultiHeadAttention,
    Input
)

from tensorflow.keras.callbacks import EarlyStopping

# ==========================================================
# CONFIG
# ==========================================================

SEQ_LEN = 60

TRAIN_SPLIT = 0.8

EPOCHS = 30

BATCH_SIZE = 32

DATA_PATH = r"C:\Users\Niruban\Documents\Suriya\HCLGUVI\Project_5\Bitcoin Historical Data.csv"

FEATURES = [
    "Price",
    "Open",
    "High",
    "Low",
    "Vol.",
    "Change %"
]

# ==========================================================
# LOAD DATA
# ==========================================================

def load_data(path):

    df = pd.read_csv(path)

    df.columns = [c.strip() for c in df.columns]

    # ------------------------------------------------------
    # DATE
    # ------------------------------------------------------

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date")

    # ------------------------------------------------------
    # CLEAN NUMERIC COLUMNS
    # ------------------------------------------------------

    numeric_cols = [
        "Price",
        "Open",
        "High",
        "Low"
    ]

    for col in numeric_cols:

        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

    # ------------------------------------------------------
    # CLEAN VOLUME
    # ------------------------------------------------------

    def clean_volume(x):

        x = str(x).replace(",", "")

        if "K" in x:
            return float(x.replace("K", "")) * 1000

        elif "M" in x:
            return float(x.replace("M", "")) * 1000000

        elif "B" in x:
            return float(x.replace("B", "")) * 1000000000

        else:
            try:
                return float(x)
            except:
                return 0

    df["Vol."] = df["Vol."].apply(clean_volume)

    # ------------------------------------------------------
    # CLEAN CHANGE %
    # ------------------------------------------------------

    df["Change %"] = (
        df["Change %"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .astype(float)
    )

    return df


# ==========================================================
# PREPROCESS
# ==========================================================

def preprocess_data(df):

    feature_scaler = MinMaxScaler()

    scaled_features = feature_scaler.fit_transform(
        df[FEATURES]
    )

    target_scaler = MinMaxScaler()

    target_scaler.fit(
        df[["Price"]]
    )

    return scaled_features, target_scaler


# ==========================================================
# CREATE SEQUENCES
# ==========================================================

def create_sequences(data, seq_len, horizon):

    X = []
    y = []

    for i in range(
        len(data) - seq_len - horizon
    ):

        X.append(
            data[i:i + seq_len]
        )

        y.append(
            data[i + seq_len + horizon - 1][0]
        )

    return np.array(X), np.array(y)


# ==========================================================
# SPLIT DATA
# ==========================================================

def split_data(X, y):

    split_index = int(
        len(X) * TRAIN_SPLIT
    )

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ==========================================================
# CNN MODEL
# ==========================================================

def build_cnn_model(input_shape):

    model = Sequential([

        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            input_shape=input_shape
        ),

        Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu"
        ),

        GlobalAveragePooling1D(),

        Dense(64, activation="relu"),

        Dropout(0.2),

        Dense(1)

    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


# ==========================================================
# RNN MODEL
# ==========================================================

def build_rnn_model(input_shape):

    model = Sequential([

        SimpleRNN(
            64,
            return_sequences=True,
            input_shape=input_shape
        ),

        Dropout(0.2),

        SimpleRNN(32),

        Dense(1)

    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


# ==========================================================
# LSTM MODEL
# ==========================================================

def build_lstm_model(input_shape):

    model = Sequential([

        LSTM(
            64,
            return_sequences=True,
            input_shape=input_shape
        ),

        Dropout(0.2),

        LSTM(32),

        Dense(1)

    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


# ==========================================================
# TRANSFORMER MODEL
# ==========================================================

def transformer_encoder(
    inputs,
    head_size=32,
    num_heads=2,
    ff_dim=64,
    dropout=0.1
):

    x = MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(inputs, inputs)

    x = Dropout(dropout)(x)

    x = LayerNormalization(
        epsilon=1e-6
    )(x)

    res = x + inputs

    x = Dense(
        ff_dim,
        activation="relu"
    )(res)

    x = Dropout(dropout)(x)

    x = Dense(
        inputs.shape[-1]
    )(x)

    return LayerNormalization(
        epsilon=1e-6
    )(x + res)


def build_transformer_model(input_shape):

    inputs = Input(shape=input_shape)

    x = transformer_encoder(inputs)

    x = GlobalAveragePooling1D()(x)

    x = Dense(64, activation="relu")(x)

    x = Dropout(0.2)(x)

    outputs = Dense(1)(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


# ==========================================================
# EVALUATION
# ==========================================================

def evaluate_forecast(
    model,
    X_test,
    y_test,
    target_scaler,
    title
):

    preds = model.predict(X_test)

    # ------------------------------------------------------
    # INVERSE TRANSFORM
    # ------------------------------------------------------

    preds_actual = target_scaler.inverse_transform(
        preds
    ).flatten()

    actual_actual = target_scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    # ------------------------------------------------------
    # METRICS
    # ------------------------------------------------------

    mae = mean_absolute_error(
        actual_actual,
        preds_actual
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_actual,
            preds_actual
        )
    )

    mape = mean_absolute_percentage_error(
        actual_actual,
        preds_actual
    ) * 100

    r2 = r2_score(
        actual_actual,
        preds_actual
    )

    # ------------------------------------------------------
    # PRINT METRICS
    # ------------------------------------------------------

    print(f"\n{title}")

    print(f"MAE  : {mae:.2f}")

    print(f"RMSE : {rmse:.2f}")

    print(f"MAPE : {mape:.2f}%")

    print(f"R2   : {r2:.4f}")

    return (
        actual_actual,
        preds_actual,
        mae,
        rmse,
        mape,
        r2
    )


# ==========================================================
# FORECAST VISUALIZATION
# ==========================================================

def plot_forecast(
    actual,
    predicted,
    title
):

    plt.figure(figsize=(14, 6))

    plt.plot(
        actual,
        label="Actual Price"
    )

    plt.plot(
        predicted,
        label="Predicted Price"
    )

    plt.title(title)

    plt.xlabel("Time")

    plt.ylabel("Bitcoin Price")

    plt.legend()

    plt.grid(True)

    plt.show()


# ==========================================================
# ERROR DISTRIBUTION
# ==========================================================

def plot_error_distribution(
    actual,
    predicted,
    title
):

    errors = actual - predicted

    plt.figure(figsize=(10, 5))

    plt.hist(
        errors,
        bins=30
    )

    plt.title(title)

    plt.xlabel("Prediction Error")

    plt.ylabel("Frequency")

    plt.grid(True)

    plt.show()


# ==========================================================
# EARLY STOPPING
# ==========================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ==========================================================
# MAIN
# ==========================================================

def main():

    # ------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------

    df = load_data(DATA_PATH)

    # ------------------------------------------------------
    # PREPROCESS
    # ------------------------------------------------------

    scaled_data, target_scaler = preprocess_data(df)

    # ======================================================
    # CREATE DATASETS
    # ======================================================

    X_1d, y_1d = create_sequences(scaled_data,SEQ_LEN,1)

    X_3d, y_3d = create_sequences(scaled_data,SEQ_LEN, 3)

    X_7d, y_7d = create_sequences(scaled_data,SEQ_LEN,7)

    # ======================================================
    # SPLIT DATA
    # ======================================================

    X_train_1d, X_test_1d, y_train_1d, y_test_1d = split_data(X_1d, y_1d)

    X_train_3d, X_test_3d, y_train_3d, y_test_3d = split_data(X_3d, y_3d)

    X_train_7d, X_test_7d, y_train_7d, y_test_7d = split_data(X_7d, y_7d)

    # ======================================================
    # MODEL BUILDERS
    # ======================================================

    models = {

        "CNN": build_cnn_model,

        "RNN": build_rnn_model,

        "LSTM": build_lstm_model,

        "Transformer": build_transformer_model
    }

    # ======================================================
    # FINAL COMPARISON STORAGE
    # ======================================================

    comparison_results = []

    # ======================================================
    # HORIZON DATA
    # ======================================================

    horizon_data = {

        "1-Day": (
            X_train_1d,
            X_test_1d,
            y_train_1d,
            y_test_1d
        ),

        "3-Day": (
            X_train_3d,
            X_test_3d,
            y_train_3d,
            y_test_3d
        ),

        "7-Day": (
            X_train_7d,
            X_test_7d,
            y_train_7d,
            y_test_7d
        )
    }

    # ======================================================
    # TRAIN ALL MODELS
    # ======================================================

    for model_name, builder in models.items():

        for horizon_name, data_tuple in horizon_data.items():

            print(
                f"\n================================="
            )

            print(
                f"{model_name} - {horizon_name}"
            )

            print(
                f"================================="
            )

            X_train, X_test, y_train, y_test = data_tuple

            # --------------------------------------------------
            # BUILD MODEL
            # --------------------------------------------------

            model = builder(
                (
                    X_train.shape[1],
                    X_train.shape[2]
                )
            )

            # --------------------------------------------------
            # TRAIN MODEL
            # --------------------------------------------------

            history = model.fit(

                X_train,
                y_train,

                validation_data=(
                    X_test,
                    y_test
                ),

                epochs=EPOCHS,

                batch_size=BATCH_SIZE,

                callbacks=[early_stop],

                verbose=1
            )

            # --------------------------------------------------
            # SAVE MODEL
            # --------------------------------------------------

            save_name = (
                f"{model_name.lower()}_"
                f"{horizon_name.replace('-', '').lower()}.h5"
            )

            model.save(save_name)

            # --------------------------------------------------
            # EVALUATE
            # --------------------------------------------------

            actuals, preds, mae, rmse, mape, r2 = evaluate_forecast(

                model,

                X_test,

                y_test,

                target_scaler,

                f"{model_name} {horizon_name}"
            )

            # --------------------------------------------------
            # FORECAST PLOT
            # --------------------------------------------------

            plot_forecast(

                actuals,

                preds,

                f"{model_name} {horizon_name} Forecast"
            )

            # --------------------------------------------------
            # ERROR DISTRIBUTION
            # --------------------------------------------------

            plot_error_distribution(

                actuals,

                preds,

                f"{model_name} {horizon_name} Errors"
            )

            # --------------------------------------------------
            # SAVE METRICS
            # --------------------------------------------------

            comparison_results.append({

                "Model": model_name,

                "Horizon": horizon_name,

                "MAE": round(mae, 2),

                "RMSE": round(rmse, 2),

                "MAPE": round(mape, 2),

                "R2": round(r2, 4)

            })

    # ======================================================
    # FINAL COMPARISON
    # ======================================================

    comparison_df = pd.DataFrame(
        comparison_results
    )

    print(
        "\n================================="
    )

    print(
        "FINAL MODEL COMPARISON"
    )

    print(
        "================================="
    )

    print(comparison_df)

    # ------------------------------------------------------
    # SAVE CSV
    # ------------------------------------------------------

    comparison_df.to_csv(

        "all_model_comparison.csv",

        index=False
    )


# ==========================================================
# START
# ==========================================================

if __name__ == "__main__":

    main()