# ==========================================================
# CryptoCast - PyTorch Production Dashboard Version
# CNN | RNN | LSTM | Transformer
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

# ==========================================================
# CONFIG
# ==========================================================

SEQ_LEN = 60
EPOCHS = 30
BATCH_SIZE = 32
LR = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = r"C:\Users\Niruban\Documents\Suriya\HCLGUVI\Project_5\Bitcoin Historical Data.csv"

FEATURES = ["Price", "Open", "High", "Low", "Vol.", "Change %"]

HORIZONS = {"1d": 1, "3d": 3, "7d": 7}

# ==========================================================
# LOAD DATA
# ==========================================================

def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    numeric_cols = ["Price", "Open", "High", "Low"]

    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def clean_volume(x):
        try:
            x = str(x).replace(",", "")
            if "K" in x: return float(x.replace("K", "")) * 1e3
            if "M" in x: return float(x.replace("M", "")) * 1e6
            if "B" in x: return float(x.replace("B", "")) * 1e9
            return float(x)
        except:
            return np.nan

    df["Vol."] = df["Vol."].apply(clean_volume)

    df["Change %"] = df["Change %"].astype(str).str.replace("%", "", regex=False)
    df["Change %"] = pd.to_numeric(df["Change %"], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    return df

# ==========================================================
# PREPROCESS
# ==========================================================

def preprocess(df):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df[FEATURES])

    with open("feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return data

# ==========================================================
# SEQUENCES
# ==========================================================

def create_sequences(data, seq_len, horizon):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len+horizon-1][0])

    return np.array(X), np.array(y)

# ==========================================================
# DATASET
# ==========================================================

class CryptoDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ==========================================================
# MODELS
# ==========================================================

class CNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, 3),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3),
            nn.ReLU()
        )
        self.fc = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = torch.mean(x, dim=2)
        return self.fc(x)

class RNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])

class LSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

class Transformer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embed = nn.Linear(input_dim, 64)

        enc = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, 2)

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        x = x[:, -1]
        return self.fc(x)

# ==========================================================
# TRAIN
# ==========================================================

def train(model, loader):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    model.train()

    for epoch in range(EPOCHS):
        total = 0

        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            opt.zero_grad()
            pred = model(X).squeeze()
            loss = loss_fn(pred, y)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += loss.item()

        print(f"Epoch {epoch+1}: {total/len(loader):.6f}")

    return model

# ==========================================================
# EVALUATION
# ==========================================================

def evaluate(model, loader):
    model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            out = model(X).cpu().numpy().flatten()

            preds.extend(out)
            actuals.extend(y.numpy())

    preds = np.nan_to_num(np.array(preds))
    actuals = np.nan_to_num(np.array(actuals))

    #return actuals, preds
    return np.array(actuals), np.array(preds)

# ==========================================================
# METRICS
# ==========================================================

def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "R2": r2_score(y_true, y_pred)
    }

# ==========================================================
# PLOT - PRICE + ERROR
# ==========================================================

def plot_results(all_results, all_errors):
    models = ["CNN", "RNN", "LSTM", "Transformer"]
    horizons = ["1d", "3d", "7d"]

    # PRICE PLOTS
    fig, axes = plt.subplots(4, 3, figsize=(18, 12))
    fig.suptitle("Bitcoin Price Prediction", fontsize=16)

    for i, m in enumerate(models):
        for j, h in enumerate(horizons):
            if (m, h) in all_results:
                ax = axes[i][j]
                a, p = all_results[(m, h)]
                ax.plot(a, label="Actual")
                ax.plot(p, label="Pred")
                ax.set_title(f"{m}-{h}")
                ax.grid(True)

    plt.tight_layout()
    plt.show()

    # ERROR PLOTS
    fig, axes = plt.subplots(4, 3, figsize=(18, 12))
    fig.suptitle("Prediction Error Distribution", fontsize=16)

    for i, m in enumerate(models):
        for j, h in enumerate(horizons):
            if (m, h) in all_errors:
                ax = axes[i][j]
                err = all_errors[(m, h)]
                ax.hist(err, bins=20)
                ax.set_title(f"{m}-{h} Error")

    plt.tight_layout()
    plt.show()

# ==========================================================
# MAIN
# ==========================================================

def main():

    df = load_data(DATA_PATH)
    data = preprocess(df)

    results = {}
    errors = {}
    metrics_list = []

    for h_name, h in HORIZONS.items():

        X, y = create_sequences(data, SEQ_LEN, h)

        split = int(len(X) * 0.8)

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        train_loader = DataLoader(CryptoDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(CryptoDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

        models = {
            "CNN": CNN,
            "RNN": RNN,
            "LSTM": LSTM,
            "Transformer": Transformer
        }

        for name, ModelClass in models.items():

            print(f"\nTraining {name} - {h_name}")

            model = ModelClass(len(FEATURES))
            model = train(model, train_loader)

            y_true, y_pred = evaluate(model, test_loader)

            m = metrics(y_true, y_pred)

            print(name, h_name, m)

            results[(name, h_name)] = (y_true, y_pred)

            y_true = np.array(y_true).flatten()
            y_pred = np.array(y_pred).flatten()

            print(type(y_true), type(y_pred))
            print(np.shape(y_true), np.shape(y_pred))

            errors[(name, h_name)] = np.array(y_true) - np.array(y_pred)
            
            metrics_list.append({
                "Model": name,
                "Horizon": h_name,
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "MAPE": m["MAPE"],
                "R2": m["R2"]
            })

            torch.save(model.state_dict(), f"{name}_{h_name}.pt")

            with open(f"{name}_{h_name}.pkl", "wb") as f:
                pickle.dump((y_true, y_pred), f)

    # SAVE METRICS FOR STREAMLIT
    pd.DataFrame(metrics_list).to_csv("metrics_summary.csv", index=False)

    plot_results(results, errors)

    print("DONE - Ready for Streamlit Dashboard 🚀")

# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()