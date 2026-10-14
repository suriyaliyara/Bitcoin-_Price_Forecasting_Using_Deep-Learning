import os
import pickle
import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")                      # save plots to files (no pop-up windows)
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
SEQ_LEN = 60                               # past 60 days as input

HORIZONS = {
    "1d": 1,
    "3d": 3,
    "7d": 7
}

MODEL_NAMES = ["CNN", "RNN", "LSTM", "Transformer"]

START_DATE = "2017-01-01"                  # None = use all history from 2010
TRAIN_SPLIT = 0.70                         # oldest 70% -> train
VAL_SPLIT = 0.15                           # next 15%  -> validation (early stopping)
                                           # newest 15% -> test

# "log_return" (recommended): predict % move from today, then convert back to price
# "price": predict the scaled price directly (fails when test prices go above the train max)
TARGET_MODE = "log_return"

BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10                              # early stopping: stop after 10 epochs without improvement
LR = 1e-3
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Bitcoin Historical Data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# DATA LOADER
# =========================
def load_data(path):

    df = pd.read_csv(path, encoding="utf-8-sig")       # file starts with a BOM

    df.columns = [c.strip() for c in df.columns]

    # Convert date (file is DD-MM-YYYY, newest first)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

    # Sort by date (oldest first) and remove duplicates
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

    # =========================
    # CLEAN NUMERIC COLUMNS
    # =========================

    numeric_cols = ["Price", "Open", "High", "Low"]

    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype(float)
        )
        df.loc[df[col] <= 0, col] = np.nan              # a few zero prices in 2010

    # -------------------------
    # CLEAN VOLUME COLUMN
    # -------------------------
    def clean_volume(x):

        x = str(x).replace(",", "").strip()

        if x.endswith("K"):
            return float(x[:-1]) * 1_000

        elif x.endswith("M"):
            return float(x[:-1]) * 1_000_000

        elif x.endswith("B"):
            return float(x[:-1]) * 1_000_000_000

        else:
            try:
                return float(x)
            except ValueError:
                return np.nan

    df["Vol."] = df["Vol."].apply(clean_volume)

    # -------------------------
    # CLEAN CHANGE %
    # -------------------------
    df["Change %"] = (
        df["Change %"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .astype(float)
    )

    # -------------------------
    # HANDLE MISSING VALUES
    # -------------------------
    # crypto trades every day -> full daily calendar, forward-fill gaps
    df = df.set_index("Date")
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))
    df.index.name = "Date"
    df = df.ffill().bfill()

    if START_DATE:
        df = df.loc[START_DATE:]

    df = df.reset_index()

    print(f"Loaded {len(df)} daily rows: {df['Date'].iloc[0].date()} -> {df['Date'].iloc[-1].date()}")

    return df


# =========================
# FEATURES
# =========================
def add_features(df):
    """
    Scale-free features: they look the same whether Bitcoin is $10K or $70K.
    Open is not used: in this data Open = previous day's close (duplicates log_ret).
    Volume is not used: the source mixes K/M/B units with a level break.
    """
    df = df.copy()
    df["log_ret"] = np.log(df["Price"]).diff()                              # daily return
    df["hl_range"] = (df["High"] - df["Low"]) / df["Price"]                 # intraday swing
    df["vol_7d"] = df["log_ret"].rolling(7).std()                           # recent volatility
    df["ma_gap_20"] = np.log(df["Price"] / df["Price"].rolling(20).mean())  # gap to 20-day average
    df = df.dropna().reset_index(drop=True)                                 # first 19 rows dropped
    return df


FEATURES = ["log_ret", "hl_range", "vol_7d", "ma_gap_20"]
if TARGET_MODE == "price":
    FEATURES = ["Price"] + FEATURES


# =========================
# TIME-BASED SPLIT
# =========================
def time_split(n, horizon):
    """
    Split the forecast days ("origins") by time, no shuffling.
    A sample belongs to a block only if ALL its target days are inside that block,
    so no future information leaks into training.
    """
    train_end = int(n * TRAIN_SPLIT) - 1
    val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT)) - 1

    origins = np.arange(SEQ_LEN - 1, n - horizon)          # every possible "today"

    train_idx = origins[origins + horizon <= train_end]
    val_idx = origins[(origins >= train_end) & (origins + horizon <= val_end)]
    test_idx = origins[origins >= val_end]

    return train_idx, val_idx, test_idx, train_end


# =========================
# PREPROCESSOR
# =========================
class Preprocessor:
    """Scalers are fitted on TRAINING rows only, then applied to val/test (no leakage)."""

    def __init__(self, features, target_mode):
        self.features = features
        self.target_mode = target_mode
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler() if target_mode == "price" else StandardScaler()

    def fit_transform_features(self, df, train_end):
        self.x_scaler.fit(df[self.features].iloc[: train_end + 1])
        return self.x_scaler.transform(df[self.features])

    def make_target(self, close, horizon):
        """target[t] = price h days after t (price mode) or log(P[t+h] / P[t]) (return mode)."""
        future = np.full(len(close), np.nan)
        future[:-horizon] = close[horizon:]
        if self.target_mode == "price":
            return future
        return np.log(future / close)

    def fit_transform_target(self, target, train_idx):
        self.y_scaler.fit(target[train_idx].reshape(-1, 1))
        return self.y_scaler.transform(target.reshape(-1, 1)).ravel()

    def to_price(self, y_scaled, close_today):
        """Turn scaled model output back into USD."""
        y = self.y_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()
        if self.target_mode == "price":
            return y
        return close_today * np.exp(y)


# =========================
# DATASET (sliding window)
# =========================
class CryptoDataset(Dataset):
    """
    Sliding window: for each forecast day t,
      X = feature rows [t-59 ... t]   (60 days)
      y = target for day t            (price h days later)
    """

    def __init__(self, data, target, origins, seq_len):
        self.X, self.y = self.create_sequences(data, target, origins, seq_len)

    def create_sequences(self, data, target, origins, seq_len):
        X = np.stack([data[t - seq_len + 1: t + 1] for t in origins])
        y = target[origins]

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# MODELS
# =========================

# ----- CNN -----
class CNNModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, 3),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)             # (batch, features, days) for Conv1d
        x = self.conv(x)
        x = torch.mean(x, dim=2)           # average over time
        return self.fc(x)


# ----- RNN -----
class RNNModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim, 64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(self.dropout(out[:, -1, :]))      # last day's hidden state


# ----- LSTM -----
class LSTMModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


# ----- TRANSFORMER -----
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, seq_len=SEQ_LEN):
        super().__init__()

        self.embedding = nn.Linear(input_dim, d_model)

        # attention has no sense of order -> learned position embedding for each of the 60 days
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.fc(x)


# =========================
# TRAINING
# =========================
def get_model(name, input_dim):
    if name == "CNN":
        return CNNModel(input_dim).to(DEVICE)
    if name == "RNN":
        return RNNModel(input_dim).to(DEVICE)
    if name == "LSTM":
        return LSTMModel(input_dim).to(DEVICE)
    if name == "Transformer":
        return TransformerModel(input_dim).to(DEVICE)
    raise ValueError(name)


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total = 0.0

    with torch.set_grad_enabled(training):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = model(X).squeeze(-1)
            loss = criterion(preds, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(X)

    return total / len(loader.dataset)


def train_model(model, train_loader, val_loader):
    criterion = nn.HuberLoss()                  # robust to crash-day outliers
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val, best_state, wait = float("inf"), None, 0
    history = {"train": [], "val": []}

    for epoch in range(EPOCHS):
        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        val_loss = run_epoch(model, val_loader, criterion)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        # -------- early stopping --------
        if val_loss < best_val:
            best_val, wait = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)          # keep the best epoch, not the last one
    print(f"   stopped at epoch {epoch + 1}, best val loss {best_val:.4f}")

    return model, history


# =========================
# EVALUATION
# =========================
def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for X, _ in loader:
            preds.extend(model(X.to(DEVICE)).cpu().numpy().ravel())
    return np.array(preds)


def compute_metrics(actual, pred, close_today):
    """All metrics in real USD, not scaled values."""
    actual, pred = np.asarray(actual), np.asarray(pred)
    is_naive = np.allclose(pred, close_today)
    return {
        "MAE": mean_absolute_error(actual, pred),
        "RMSE": np.sqrt(mean_squared_error(actual, pred)),
        "MAPE": np.mean(np.abs((actual - pred) / actual)) * 100,
        "R2": r2_score(actual, pred),
        # up/down correct vs today (not defined for naive, which never moves)
        "DirAcc": np.nan if is_naive else
        np.mean(np.sign(pred - close_today) == np.sign(actual - close_today)) * 100,
    }


# =========================
# VISUALIZATION
# =========================
def plot_all_models(all_results, out_dir):

    fig, axes = plt.subplots(len(MODEL_NAMES), len(HORIZONS), figsize=(20, 16), sharex="col")

    fig.suptitle("Bitcoin Forecast vs Actual (test period, USD)", fontsize=20)

    for i, model in enumerate(MODEL_NAMES):
        for j, horizon in enumerate(HORIZONS):
            r = all_results[(model, horizon)]
            ax = axes[i, j]
            ax.plot(r["dates"], r["y_true"], label="Actual", color="black", lw=1.5)
            ax.plot(r["dates"], r["y_pred"], label=model, lw=1)
            ax.plot(r["dates"], r["naive"], label="Naive", color="gray", lw=0.8, ls="--")
            ax.set_title(f"{model} - {horizon}")
            ax.grid(True)
            if i == 0 and j == 0:
                ax.legend()

    fig.autofmt_xdate()
    plt.tight_layout()
    path = os.path.join(out_dir, "forecast_grid.png")
    plt.savefig(path, dpi=110)
    plt.close(fig)
    print(f"Saved {path}")


def plot_loss_curves(histories, out_dir):
    fig, axes = plt.subplots(len(HORIZONS), len(MODEL_NAMES), figsize=(18, 10))
    for i, horizon in enumerate(HORIZONS):
        for j, model in enumerate(MODEL_NAMES):
            h = histories[(model, horizon)]
            ax = axes[i, j]
            ax.plot(h["train"], label="train")
            ax.plot(h["val"], label="val")
            ax.set_title(f"{model} - {horizon}")
            ax.grid(True)
    axes[0, 0].legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "loss_curves.png")
    plt.savefig(path, dpi=110)
    plt.close(fig)
    print(f"Saved {path}")


# =========================
# MAIN PIPELINE
# =========================
def main():

    set_seed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)

    df = add_features(load_data(DATA_PATH))
    close = df["Price"].values
    dates = df["Date"].values
    n = len(df)

    # ==========================================
    # STORE ALL RESULTS
    # ==========================================
    all_results = {}
    histories = {}
    metric_rows = []
    comparison_frames = []
    next_forecast_rows = []

    for horizon_name, horizon in HORIZONS.items():

        print(f"\n===== Horizon {horizon_name} (price {horizon} day(s) ahead) =====")

        # --------------------------------------
        # TIME SERIES SPLIT + SCALING (train only)
        # --------------------------------------
        train_idx, val_idx, test_idx, train_end = time_split(n, horizon)

        prep = Preprocessor(FEATURES, TARGET_MODE)
        data = prep.fit_transform_features(df, train_end)
        target_raw = prep.make_target(close, horizon)
        target = prep.fit_transform_target(target_raw, train_idx)

        print(f"Train: {len(train_idx)} samples ({pd.Timestamp(dates[train_idx[0]]).date()} -> {pd.Timestamp(dates[train_idx[-1]]).date()})")
        print(f"Val:   {len(val_idx)} samples ({pd.Timestamp(dates[val_idx[0]]).date()} -> {pd.Timestamp(dates[val_idx[-1]]).date()})")
        print(f"Test:  {len(test_idx)} samples ({pd.Timestamp(dates[test_idx[0]]).date()} -> {pd.Timestamp(dates[test_idx[-1]]).date()})")

        train_loader = DataLoader(CryptoDataset(data, target, train_idx, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False)
        val_loader = DataLoader(CryptoDataset(data, target, val_idx, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(CryptoDataset(data, target, test_idx, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False)

        # true future prices, today's prices, and the dates being predicted
        close_today = close[test_idx]
        y_true = close[test_idx + horizon]
        target_dates = pd.to_datetime(dates[test_idx + horizon])

        # --------------------------------------
        # NAIVE BASELINE: "price in h days = today's price"
        # --------------------------------------
        naive = close_today.copy()
        m = compute_metrics(y_true, naive, close_today)
        metric_rows.append({"Model": "Naive", "Horizon": horizon_name, **m})
        naive_rmse = m["RMSE"]
        print(f"NAIVE        MAE ${m['MAE']:,.0f} | RMSE ${m['RMSE']:,.0f} | MAPE {m['MAPE']:.2f}%")

        comparison = pd.DataFrame({
            "Forecast_Date": pd.to_datetime(dates[test_idx]).date,
            "Target_Date": target_dates.date,
            "Horizon": horizon_name,
            "Today_Close": close_today.round(2),
            "Actual": y_true.round(2),
            "Naive": naive.round(2),
        })

        # ==========================================
        # MODELS
        # ==========================================
        for model_name in MODEL_NAMES:

            print(f"\nTraining {model_name} for {horizon_name}")
            set_seed()                                     # same start for every model = fair comparison

            model = get_model(model_name, len(FEATURES))
            model, history = train_model(model, train_loader, val_loader)
            histories[(model_name, horizon_name)] = history

            # --------------------------------------
            # EVALUATE (in USD)
            # --------------------------------------
            y_pred = prep.to_price(predict(model, test_loader), close_today)
            m = compute_metrics(y_true, y_pred, close_today)
            m["Skill_vs_Naive_%"] = (1 - m["RMSE"] / naive_rmse) * 100
            metric_rows.append({"Model": model_name, "Horizon": horizon_name, **m})

            print(
                f"{model_name:<12} MAE ${m['MAE']:,.0f} | RMSE ${m['RMSE']:,.0f} | "
                f"MAPE {m['MAPE']:.2f}% | R2 {m['R2']:.3f} | DirAcc {m['DirAcc']:.1f}% | "
                f"vs Naive {m['Skill_vs_Naive_%']:+.1f}%"
            )

            comparison[model_name] = y_pred.round(2)

            # --------------------------------------
            # STORE RESULTS (dashboard reads these)
            # --------------------------------------
            result = {
                "dates": target_dates,
                "y_true": y_true,
                "y_pred": y_pred,
                "naive": naive,
                "close_today": close_today,
            }
            all_results[(model_name, horizon_name)] = result
            with open(os.path.join(OUTPUT_DIR, f"{model_name}_{horizon_name}.pkl"), "wb") as f:
                pickle.dump(result, f)
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "models", f"{model_name}_{horizon_name}.pt"))

            # --------------------------------------
            # NEXT FORECAST from the latest 60 days
            # --------------------------------------
            last = n - 1
            x_last = torch.tensor(data[last - SEQ_LEN + 1: last + 1][None], dtype=torch.float32).to(DEVICE)
            model.eval()
            with torch.no_grad():
                p = prep.to_price(model(x_last).cpu().numpy().ravel(), close[last])[0]
            next_forecast_rows.append({
                "Model": model_name, "Horizon": horizon_name,
                "Last_Date": pd.Timestamp(dates[last]).date(), "Last_Close": round(close[last], 2),
                "Target_Date": (pd.Timestamp(dates[last]) + pd.Timedelta(days=horizon)).date(),
                "Forecast": round(p, 2), "Change_%": round((p / close[last] - 1) * 100, 2),
            })

        comparison_frames.append(comparison)

        # show the last 7 forecasts side by side
        print(f"\nLast 7 test forecasts, {horizon_name} (USD):")
        cols = ["Target_Date", "Actual", "Naive"] + MODEL_NAMES
        print(comparison[cols].tail(7).to_string(index=False, float_format=lambda v: f"{v:,.0f}"))

    # ==========================================
    # SAVE SUMMARY FILES
    # ==========================================
    metrics_df = pd.DataFrame(metric_rows).round(4)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)
    pd.concat(comparison_frames).to_csv(os.path.join(OUTPUT_DIR, "price_comparison.csv"), index=False)
    pd.DataFrame(next_forecast_rows).to_csv(os.path.join(OUTPUT_DIR, "next_forecast.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "histories.pkl"), "wb") as f:
        pickle.dump(histories, f)

    print("\n===== MAPE % by model and horizon (lower is better) =====")
    print(metrics_df.pivot(index="Model", columns="Horizon", values="MAPE").loc[["Naive"] + MODEL_NAMES].round(2))
    print("\n===== RMSE skill vs Naive % (positive = model beats 'price stays the same') =====")
    print(metrics_df.pivot(index="Model", columns="Horizon", values="Skill_vs_Naive_%").loc[MODEL_NAMES].round(1))

    plot_all_models(all_results, OUTPUT_DIR)
    plot_loss_curves(histories, OUTPUT_DIR)
    print(f"\nAll outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
