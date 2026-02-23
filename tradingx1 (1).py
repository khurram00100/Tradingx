import os
import time
import sqlite3
import threading
import queue
import json
import pickle
import math
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import requests
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Try to import speech recognition (optional)
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except:
    SPEECH_AVAILABLE = False

# Try to import pyttsx3 for text-to-speech (optional)
try:
    import pyttsx3
    TTS_AVAILABLE = True
except:
    TTS_AVAILABLE = False

# ==========================
# CONFIGURATION
# ==========================
DATA_DIR = "jarvis_ann_pro_data"
MODELS_DIR = os.path.join(DATA_DIR, "models")
SCALERS_DIR = os.path.join(DATA_DIR, "scalers")
METRICS_DIR = os.path.join(DATA_DIR, "metrics")
CHAT_HISTORY_PATH = os.path.join(DATA_DIR, "chat_history.json")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")
LOG_PATH = os.path.join(DATA_DIR, "app.log")

for d in [DATA_DIR, MODELS_DIR, SCALERS_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)

# Thread-safe logging
log_q = queue.Queue()
file_log_lock = threading.Lock()

def log(msg: str, level: str = "INFO"):
    """Thread-safe logging"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] [{level}] {msg}"
    print(formatted)
    log_q.put(formatted)
    
    with file_log_lock:
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(formatted + "\n")
        except:
            pass

# ==========================
# DATABASE SETUP
# ==========================
DB_PATH = os.path.join(DATA_DIR, "jarvis_data.db")

class DatabaseManager:
    """Secure local database for user data"""
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.lock = threading.Lock()
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with self.lock:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset TEXT,
                    price REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset TEXT,
                    current_price REAL,
                    predicted_price REAL,
                    prediction_hours INTEGER,
                    confidence REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender TEXT,
                    message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset TEXT,
                    score REAL,
                    rmse REAL,
                    mae REAL,
                    r2 REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.conn.commit()
    
    def save_setting(self, key: str, value: str):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO user_settings (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (key, value))
            self.conn.commit()
    
    def get_setting(self, key: str, default=None):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('SELECT value FROM user_settings WHERE key = ?', (key,))
            result = cursor.fetchone()
            return result[0] if result else default
    
    def save_price(self, asset: str, price: float):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO price_history (asset, price)
                VALUES (?, ?)
            ''', (asset, price))
            self.conn.commit()
    
    def save_prediction(self, asset: str, current_price: float, predicted_price: float, 
                       prediction_hours: int, confidence: float):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (asset, current_price, predicted_price, 
                                       prediction_hours, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (asset, current_price, predicted_price, prediction_hours, confidence))
            self.conn.commit()
    
    def save_chat_message(self, sender: str, message: str):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO chat_history (sender, message)
                VALUES (?, ?)
            ''', (sender, message))
            self.conn.commit()
    
    def get_chat_history(self, limit: int = 50):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT sender, message, timestamp 
                FROM chat_history 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            return cursor.fetchall()
    
    def save_metrics(self, asset: str, score: float, rmse: float, mae: float, r2: float):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO model_metrics (asset, score, rmse, mae, r2)
                VALUES (?, ?, ?, ?, ?)
            ''', (asset, score, rmse, mae, r2))
            self.conn.commit()
    
    def get_price_history(self, asset: str, hours: int = 168):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT price, timestamp 
                FROM price_history 
                WHERE asset = ? 
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp
            ''', (asset, hours))
            return cursor.fetchall()
    
    def close(self):
        self.conn.close()

# ==========================
# SETTINGS
# ==========================
DEFAULT_SETTINGS = {
    "fetch_interval_seconds": 3600,
    "prediction_interval": "1hour",
    "seq_len": 24,
    "pred_hours": 6,
    "epochs_per_cycle": 15,
    "batch_size": 32,
    "learning_rate": 0.0005,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "use_binance_first": True,
    "coingecko_pro_key": "",
    "enabled_assets": {
        "BTC": True, "ETH": True, "LTC": True, "XRP": True,
        "EUR": True, "GBP": True, "JPY": True, "INR": True,
        "Gold": True
    },
    "theme_mode": "dark",
    "alert_threshold": 5.0,
    "enable_alerts": True,
    "show_confidence_bands": True
}

def load_settings() -> dict:
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                s = json.load(f)
            for k, v in DEFAULT_SETTINGS.items():
                if k not in s:
                    s[k] = v
            return s
        except:
            pass
    save_settings(DEFAULT_SETTINGS)
    return DEFAULT_SETTINGS.copy()

def save_settings(s: dict):
    try:
        db.save_setting("app_settings", json.dumps(s))
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
        log("Settings saved")
    except Exception as e:
        log(f"Settings save error: {e}", "ERROR")

db = DatabaseManager()
settings = load_settings()

# ==========================
# ASSETS
# ==========================
ASSETS = {
    "crypto": {"bitcoin": "BTC", "ethereum": "ETH", "litecoin": "LTC", "ripple": "XRP"},
    "fx": {"EUR": "EUR", "GBP": "GBP", "JPY": "JPY", "INR": "INR"},
    "gold": {"XAUUSD": "Gold"}
}

BINANCE_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "LTC": "LTCUSDT", "XRP": "XRPUSDT"}
HEADERS = {"User-Agent": "jarvis-ann-pro/3.2"}

# ==========================
# PERSISTENCE
# ==========================
def model_file(asset: str) -> str:
    return os.path.join(MODELS_DIR, f"{asset}_model.pth")

def scaler_file(asset: str) -> str:
    return os.path.join(SCALERS_DIR, f"{asset}_scaler.pkl")

def metrics_file(asset: str) -> str:
    return os.path.join(METRICS_DIR, f"{asset}_metrics.json")

def save_scaler(scaler: MinMaxScaler, asset: str):
    try:
        with open(scaler_file(asset), "wb") as f:
            pickle.dump(scaler, f)
    except:
        pass

def load_scaler(asset: str) -> Optional[MinMaxScaler]:
    path = scaler_file(asset)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except:
        return None

def save_metrics(asset: str, metrics: dict):
    try:
        metrics["timestamp"] = datetime.now().isoformat()
        with open(metrics_file(asset), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except:
        pass

def load_metrics(asset: str) -> Optional[dict]:
    path = metrics_file(asset)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

# ==========================
# NEURAL NETWORK
# ==========================
class TimeSeriesDataset(Dataset):
    def __init__(self, series: np.ndarray, seq_len: int, pred_steps: int):
        x, y = [], []
        for i in range(len(series) - seq_len - pred_steps + 1):
            x.append(series[i:i+seq_len])
            y.append(series[i+seq_len:i+seq_len+pred_steps])
        self.x = np.array(x, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return (torch.from_numpy(self.x[idx]).unsqueeze(-1), 
                torch.from_numpy(self.y[idx]))

class EnhancedLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, 
                 pred_steps=6, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, pred_steps)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# ==========================
# DATA FETCHERS
# ==========================
def fetch_crypto_binance(symbol: str, hours: int = 168) -> Optional[pd.DataFrame]:
    try:
        limit = min(1000, max(48, hours + 10))
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": "1h", "limit": limit}
        r = requests.get(url, params=params, timeout=15, headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        
        rows = []
        for k in data:
            ts = int(k[0])
            close = float(k[4])
            rows.append([pd.to_datetime(ts, unit="ms"), close])
        
        df = pd.DataFrame(rows, columns=["datetime", "price"])
        df.set_index("datetime", inplace=True)
        df = df.resample("h").ffill().tail(hours)
        return df
    except:
        return None

def fetch_crypto_coingecko(coin_id: str, hours: int = 168) -> Optional[pd.DataFrame]:
    try:
        days = max(1, (hours // 24) + 1)
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days, "interval": "hourly"}
        headers = HEADERS.copy()
        key = settings.get("coingecko_pro_key", "")
        if key:
            headers["x-cg-pro-api-key"] = key
        r = requests.get(url, params=params, timeout=20, headers=headers)
        r.raise_for_status()
        j = r.json()
        prices = j.get("prices", [])
        rows = [[pd.to_datetime(p[0], unit="ms"), float(p[1])] for p in prices]
        df = pd.DataFrame(rows, columns=["datetime", "price"])
        df.set_index("datetime", inplace=True)
        df = df.resample("h").ffill().tail(hours)
        return df
    except:
        return None

def fetch_crypto(coin_key: str, label: str, hours: int = 168) -> Optional[pd.DataFrame]:
    if settings.get("use_binance_first", True) and label in BINANCE_SYMBOLS:
        df = fetch_crypto_binance(BINANCE_SYMBOLS[label], hours=hours)
        if df is not None and not df.empty:
            return df
    return fetch_crypto_coingecko(coin_key, hours=hours)

def fetch_fx_hourly(pair_to: str = "EUR", hours: int = 168) -> Optional[pd.DataFrame]:
    try:
        end = datetime.now().date()
        start = end - timedelta(days=max(7, hours // 24 + 2))
        url = "https://api.exchangerate.host/timeseries"
        params = {"start_date": start.isoformat(), "end_date": end.isoformat(), 
                 "base": "USD", "symbols": pair_to}
        r = requests.get(url, params=params, timeout=20, headers=HEADERS)
        r.raise_for_status()
        j = r.json()
        rates = []
        for d, vals in sorted(j.get("rates", {}).items()):
            rate = vals.get(pair_to)
            if rate:
                rates.append([pd.to_datetime(d), float(rate)])
        if not rates:
            return None
        df = pd.DataFrame(rates, columns=["datetime", "price"])
        df.set_index("datetime", inplace=True)
        df = df.resample("h").interpolate(method="linear").tail(hours)
        return df
    except:
        return None

def fetch_gold_hourly(hours: int = 168) -> Optional[pd.DataFrame]:
    for symbol in ["XAUUSD=X", "GC=F"]:
        try:
            data = yf.download(tickers=symbol, period="7d", interval="60m", 
                             progress=False, threads=False)
            if data is None or data.empty:
                continue
            df = data[["Close"]].rename(columns={"Close": "price"})
            df.index = pd.to_datetime(df.index)
            df = df.resample("h").ffill().tail(hours)
            return df
        except:
            continue
    return None

# ==========================
# DATA PROCESSING
# ==========================
def clean_series(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["pct_change"] = df["price"].pct_change()
        df = df[df["pct_change"].abs() < 0.5]
        df = df.drop("pct_change", axis=1)
        df = df.dropna(subset=["price"])
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df if len(df) >= 48 else None
    except:
        return None

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    try:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        return {
            "mse": float(mse), "rmse": float(rmse), "mae": float(mae),
            "r2": float(r2), "mape": float(mape),
            "score": float(max(0, r2 * 100))
        }
    except:
        return {"score": 0.0}

# ==========================
# TRAINING & PREDICTION
# ==========================
def train_asset(asset_label: str, df_prices: pd.DataFrame, status_callback=None) -> Tuple[List[float], dict]:
    df = clean_series(df_prices)
    if df is None or len(df) < 72:
        return [], {"score": 0.0}
    
    seq_len = int(settings.get("seq_len", 24))
    pred_steps = int(settings.get("pred_hours", 6))
    epochs = int(settings.get("epochs_per_cycle", 15))
    batch_size = int(settings.get("batch_size", 32))
    lr = float(settings.get("learning_rate", 0.0005))
    hidden_size = int(settings.get("hidden_size", 128))
    num_layers = int(settings.get("num_layers", 2))
    dropout = float(settings.get("dropout", 0.2))
    
    if status_callback:
        status_callback(f"📊 Processing {asset_label} data...")
    
    series = df["price"].values.reshape(-1, 1).astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series).flatten()
    save_scaler(scaler, asset_label)
    
    dataset = TimeSeriesDataset(scaled, seq_len=seq_len, pred_steps=pred_steps)
    if len(dataset) < 10:
        return [], {"score": 0.0}
    
    val_size = max(1, int(len(dataset) * 0.15))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, train_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, val_size), shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EnhancedLSTM(1, hidden_size, num_layers, pred_steps, dropout).to(device)
    
    mpath = model_file(asset_label)
    if os.path.exists(mpath):
        try:
            model.load_state_dict(torch.load(mpath, map_location=device))
        except:
            pass
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    criterion = nn.MSELoss()
    
    train_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        if status_callback:
            status_callback(f"🧠 Training {asset_label}: Epoch {epoch+1}/{epochs}")
        
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_count += xb.size(0)
        
        avg_train_loss = train_loss / max(1, train_count)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        val_count = 0
        val_preds, val_actuals = [], []
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                output = model(xb)
                loss = criterion(output, yb)
                val_loss += loss.item() * xb.size(0)
                val_count += xb.size(0)
                val_preds.append(output.cpu().numpy())
                val_actuals.append(yb.cpu().numpy())
        
        avg_val_loss = val_loss / max(1, val_count)
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), mpath)
        else:
            patience_counter += 1
            if patience_counter >= 5:
                break
    
    if status_callback:
        status_callback(f"✅ {asset_label} training complete!")
    
    val_preds_flat = np.concatenate(val_preds).flatten()
    val_actuals_flat = np.concatenate(val_actuals).flatten()
    val_preds_orig = scaler.inverse_transform(val_preds_flat.reshape(-1, 1)).flatten()
    val_actuals_orig = scaler.inverse_transform(val_actuals_flat.reshape(-1, 1)).flatten()
    
    metrics = calculate_metrics(val_actuals_orig, val_preds_orig)
    metrics["train_loss"] = train_losses[-1] if train_losses else 0.0
    metrics["val_loss"] = avg_val_loss
    metrics["epochs_trained"] = len(train_losses)
    save_metrics(asset_label, metrics)
    
    return train_losses, metrics

def predict_asset(asset_label: str, recent_series: np.ndarray) -> Optional[np.ndarray]:
    try:
        scaler = load_scaler(asset_label)
        mpath = model_file(asset_label)
        if scaler is None or not os.path.exists(mpath):
            return None
        
        seq_len = int(settings.get("seq_len", 24))
        pred_steps = int(settings.get("pred_hours", 6))
        hidden_size = int(settings.get("hidden_size", 128))
        num_layers = int(settings.get("num_layers", 2))
        dropout = float(settings.get("dropout", 0.2))
        
        arr = np.array(recent_series, dtype=float).reshape(-1, 1)
        scaled = scaler.transform(arr).flatten()
        
        if len(scaled) < seq_len:
            return None
        
        inp = scaled[-seq_len:]
        model = EnhancedLSTM(1, hidden_size, num_layers, pred_steps, dropout)
        model.load_state_dict(torch.load(mpath, map_location="cpu"))
        model.eval()
        
        with torch.no_grad():
            
            x = torch.tensor(inp.reshape(1, seq_len, 1), dtype=torch.float32)
            preds_scaled = model(x).numpy().flatten()
        
        preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        return preds
    except:
        return None

# ==========================
# ENHANCED AI CHATBOT
# ==========================
class EnhancedANNChatbot:
    def __init__(self, app):
        self.app = app
        self.chat_history = []
        
    def process_query(self, query: str) -> str:
        query_lower = query.lower()
        
        for asset in list(ASSETS["crypto"].values()) + list(ASSETS["fx"].values()) + ["Gold"]:
            if asset.lower() in query_lower:
                return self._answer_price_query(asset, query_lower)
        
        if any(word in query_lower for word in ["predict", "forecast", "future", "will", "tomorrow", "next"]):
            return self._answer_prediction_query(query_lower)
        
        if any(word in query_lower for word in ["trend", "going up", "going down", "rising", "falling", "performance"]):
            return self._answer_trend_query(query_lower)
        
        if "best" in query_lower or "top" in query_lower or "highest" in query_lower:
            return self._answer_best_performers()
        
        if "worst" in query_lower or "bottom" in query_lower or "lowest" in query_lower:
            return self._answer_worst_performers()
        
        if "compare" in query_lower or "versus" in query_lower or "vs" in query_lower:
            return self._answer_market_analysis()
        
        return self._default_response(query)
    
    def _answer_price_query(self, asset: str, query: str) -> str:
        info = self.app.asset_state.get(asset)
        if not info:
            return f"📊 No data available for {asset}. Please wait for data fetch or click 'Fetch Now'."
        
        df = info["df"]
        current_price = df["price"].iloc[-1]
        
        changes = {}
        timeframes = [("1h", 1), ("6h", 6), ("24h", 24), ("7d", 168)]
        
        for label, hours in timeframes:
            if len(df) >= hours:
                prev_price = df["price"].iloc[-hours]
                change_pct = ((current_price - prev_price) / prev_price) * 100
                change_amt = current_price - prev_price
                changes[label] = (change_pct, change_amt)
        
        response = f"💰 **{asset} Price Analysis**\n\n"
        response += f"Current Price: ${current_price:.2f}\n\n"
        response += "**Recent Performance:**\n"
        
        for tf, (change_pct, change_amt) in changes.items():
            emoji = "📈" if change_pct > 0 else "📉"
            response += f"{emoji} {tf}: {change_pct:+.2f}% (${change_amt:+.2f})\n"
        
        preds = info.get("preds")
        if preds is not None and len(preds) > 0:
            pred_hours = len(preds)
            final_pred = preds[-1]
            pred_change_pct = ((final_pred - current_price) / current_price) * 100
            pred_change_amt = final_pred - current_price
            
            response += f"\n🔮 **{pred_hours}h Prediction:**\n"
            if pred_change_pct > 0:
                response += f"📈 BULLISH: Expected to go UP {pred_change_pct:.2f}%\n"
                response += f"   Target Price: ${final_pred:.2f} (+${pred_change_amt:.2f})\n"
            else:
                response += f"📉 BEARISH: Expected to go DOWN {abs(pred_change_pct):.2f}%\n"
                response += f"   Target Price: ${final_pred:.2f} (${pred_change_amt:.2f})\n"
            
            response += f"\n**Hour-by-Hour Forecast:**\n"
            for i, pred_price in enumerate(preds, 1):
                hour_change = ((pred_price - current_price) / current_price) * 100
                response += f"  {i}h: ${pred_price:.2f} ({hour_change:+.2f}%)\n"
        
        metrics = info.get("metrics", {})
        confidence = metrics.get("score", 0)
        response += f"\n🎯 Model Confidence: {confidence:.1f}%"
        
        return response
    
    def _answer_prediction_query(self, query: str) -> str:
        predictions = []
        for asset, info in self.app.asset_state.items():
            preds = info.get("preds")
            if preds is not None and len(preds) > 0:
                current = info["df"]["price"].iloc[-1]
                future = preds[-1]
                change_pct = ((future - current) / current) * 100
                change_amt = future - current
                confidence = info.get("metrics", {}).get("score", 0)
                predictions.append((asset, change_pct, change_amt, future, current, confidence, preds))
        
        if not predictions:
            return "🔮 I need more data to make predictions. Please wait for the data fetch cycle or click 'Fetch Now'."
        
        predictions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        pred_hours = settings.get("pred_hours", 6)
        response = f"🔮 **AI Price Predictions ({pred_hours} hours ahead)**\n\n"
        
        for asset, change_pct, change_amt, future, current, confidence, preds in predictions[:8]:
            emoji = "📈" if change_pct > 0 else "📉"
            direction = "UP" if change_pct > 0 else "DOWN"
            conf_emoji = "🟢" if confidence >= 70 else "🟡" if confidence >= 50 else "🔴"
            
            response += f"{emoji} **{asset}** - {direction} {abs(change_pct):.2f}%\n"
            response += f"   Current: ${current:.2f}\n"
            response += f"   Predicted: ${future:.2f} ({change_amt:+.2f})\n"
            response += f"   {conf_emoji} Confidence: {confidence:.1f}%\n\n"
        
        return response
    
    def _answer_trend_query(self, query: str) -> str:
        trends = []
        for asset, info in self.app.asset_state.items():
            df = info["df"]
            if len(df) < 24:
                continue
            
            current = df["price"].iloc[-1]
            prev_24h = df["price"].iloc[-24]
            trend_24h = ((current - prev_24h) / prev_24h) * 100
            
            if len(df) >= 2:
                trend_1h = ((df["price"].iloc[-1] - df["price"].iloc[-2]) / df["price"].iloc[-2]) * 100
            else:
                trend_1h = 0
            
            trends.append((asset, trend_24h, trend_1h, current))
        
        if not trends:
            return "📊 No trend data available yet."
        
        trends.sort(key=lambda x: x[1], reverse=True)
        
        response = "📊 **24-Hour Market Trends**\n\n"
        
        for asset, trend_24h, trend_1h, price in trends:
            emoji = "🟢" if trend_24h > 0 else "🔴"
            momentum = "🚀" if trend_1h > 1 else "📈" if trend_1h > 0 else "📉" if trend_1h > -1 else "💥"
            
            response += f"{emoji} **{asset}**: ${price:.2f}\n"
            response += f"   24h: {trend_24h:+.2f}% | 1h: {trend_1h:+.2f}% {momentum}\n\n"
        
        return response
    
    def _answer_best_performers(self) -> str:
        performers = []
        for asset, info in self.app.asset_state.items():
            df = info["df"]
            if len(df) < 24:
                continue
            change = ((df["price"].iloc[-1] - df["price"].iloc[-24]) / df["price"].iloc[-24]) * 100
            price = df["price"].iloc[-1]
            performers.append((asset, change, price))
        
        performers.sort(key=lambda x: x[1], reverse=True)
        
        response = "🏆 **Top Performers (24h)**\n\n"
        for i, (asset, change, price) in enumerate(performers[:5], 1):
            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i-1]
            response += f"{medal} {asset}: +{change:.2f}% (${price:.2f})\n"
        
        return response
    
    def _answer_worst_performers(self) -> str:
        performers = []
        for asset, info in self.app.asset_state.items():
            df = info["df"]
            if len(df) < 24:
                continue
            change = ((df["price"].iloc[-1] - df["price"].iloc[-24]) / df["price"].iloc[-24]) * 100
            price = df["price"].iloc[-1]
            performers.append((asset, change, price))
        
        performers.sort(key=lambda x: x[1])
        
        response = "📉 **Worst Performers (24h)**\n\n"
        for i, (asset, change, price) in enumerate(performers[:5], 1):
            response += f"{i}. {asset}: {change:.2f}% (${price:.2f})\n"
        
        return response
    
    def _answer_market_analysis(self) -> str:
        if not self.app.asset_state:
            return "📈 No market data available yet."
        
        response = "📈 **Complete Market Overview**\n\n"
        
        crypto_assets = [(a, i) for a, i in self.app.asset_state.items() if a in ["BTC", "ETH", "LTC", "XRP"]]
        fx_assets = [(a, i) for a, i in self.app.asset_state.items() if a in ["EUR", "GBP", "JPY", "INR"]]
        gold_assets = [(a, i) for a, i in self.app.asset_state.items() if a == "Gold"]
        
        if crypto_assets:
            response += "**💎 Cryptocurrency:**\n"
            for asset, info in crypto_assets:
                df = info["df"]
                if len(df) >= 24:
                    current = df["price"].iloc[-1]
                    change = ((df["price"].iloc[-1] - df["price"].iloc[-24]) / df["price"].iloc[-24]) * 100
                    emoji = "📈" if change > 0 else "📉"
                    response += f"{emoji} {asset}: ${current:.2f} ({change:+.2f}%)\n"
            response += "\n"
        
        if fx_assets:
            response += "**💱 Forex (vs USD):**\n"
            for asset, info in fx_assets:
                df = info["df"]
                if len(df) >= 24:
                    current = df["price"].iloc[-1]
                    change = ((df["price"].iloc[-1] - df["price"].iloc[-24]) / df["price"].iloc[-24]) * 100
                    emoji = "📈" if change > 0 else "📉"
                    response += f"{emoji} {asset}: {current:.4f} ({change:+.2f}%)\n"
            response += "\n"
        
        if gold_assets:
            response += "**🥇 Precious Metals:**\n"
            for asset, info in gold_assets:
                df = info["df"]
                if len(df) >= 24:
                    current = df["price"].iloc[-1]
                    change = ((df["price"].iloc[-1] - df["price"].iloc[-24]) / df["price"].iloc[-24]) * 100
                    emoji = "📈" if change > 0 else "📉"
                    response += f"{emoji} Gold: ${current:.2f} ({change:+.2f}%)\n"
        
        return response
    
    def _default_response(self, query: str) -> str:
        responses = [
            "I can help you with crypto price analysis and predictions! Try asking:\n• 'What's the price of Bitcoin?'\n• 'Predict ETH price'\n• 'Show me market trends'\n• 'Compare all assets'",
            "Ask me about specific cryptocurrencies, forex rates, or gold prices. I can also predict future prices!",
            "I'm your AI trading assistant. I can analyze prices, make predictions, and compare market performance.",
        ]
        import random
        return random.choice(responses)

# ==========================
# VOICE HANDLER
# ==========================
class VoiceHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer() if SPEECH_AVAILABLE else None
        self.tts_engine = None
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.9)
            except:
                pass
    
    def listen(self) -> Optional[str]:
        if not SPEECH_AVAILABLE or not self.recognizer:
            return None
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio)
                return text
        except:
            return None
    
    def speak(self, text: str):
        if not TTS_AVAILABLE or not self.tts_engine:
            return
        try:
            clean_text = ''.join(c for c in text if ord(c) < 128)
            self.tts_engine.say(clean_text)
            self.tts_engine.runAndWait()
        except:
            pass

# ==========================
# ASSET MANAGER
# ==========================
class AssetManager(threading.Thread):
    def __init__(self, app):
        super().__init__(daemon=True)
        self.app = app
        self._stop = threading.Event()
    
    def stop(self):
        self._stop.set()
    
    def run(self):
        log("AssetManager started", "INFO")
        
        while not self._stop.is_set():
            start_time = time.time()
            
            try:
                self.cycle_once()
            except Exception as e:
                log(f"Manager error: {e}", "ERROR")
            
            interval_map = {
                "1hour": 3600,
                "3hours": 10800,
                "6hours": 21600,
                "12hours": 43200,
                "daily": 86400
            }
            interval = interval_map.get(settings.get("prediction_interval", "1hour"), 3600)
            
            elapsed = time.time() - start_time
            sleep_time = max(60, interval - elapsed)
            
            log(f"Cycle complete in {elapsed:.1f}s. Next in {sleep_time/60:.1f}min", "INFO")
            
            for _ in range(int(sleep_time)):
                if self._stop.is_set():
                    break
                time.sleep(1)
        
        log("AssetManager stopped", "INFO")
    
    def cycle_once(self):
        hours = 168
        
        # Crypto
        for coin_id, label in ASSETS["crypto"].items():
            if not settings["enabled_assets"].get(label, True):
                continue
            try:
                self.app.update_ann_status(f"📡 Fetching {label} data from APIs...")
                df = fetch_crypto(coin_id, label, hours=hours)
                if df is None or df.empty:
                    continue
                df = clean_series(df)
                if df is None or df.empty:
                    continue
                
                self.app.update_ann_status(f"🧠 Training ANN model for {label}...")
                losses, metrics = train_asset(label, df, self.app.update_ann_status)
                
                self.app.update_ann_status(f"🔮 Generating predictions for {label}...")
                preds = predict_asset(label, df["price"].values)
                
                self.app.queue_update_asset(label, {
                    "df": df, "loss": losses, "preds": preds, "metrics": metrics
                })

                try:
                    current_price = df["price"].iloc[-1]
                    db.save_price(label, float(current_price))
    
                    if preds is not None and len(preds) > 0:
                        db.save_prediction(
                            label, 
                            float(current_price), 
                            float(preds[-1]),
                            len(preds),
                            float(metrics.get("score", 0))
                        )
    
                    db.save_metrics(
                        label,
                        float(metrics.get("score", 0)),
                        float(metrics.get("rmse", 0)),
                        float(metrics.get("mae", 0)),
                        float(metrics.get("r2", 0))
                    )    
                except Exception as e:
                    log(f"Database save error: {e}", "ERROR")
                    
                self.app.update_ann_status(f"✅ {label} analysis complete!")
            except Exception as e:
                log(f"Error processing {label}: {e}", "ERROR")
        
        # FX
        for code in ASSETS["fx"].values():
            if not settings["enabled_assets"].get(code, True):
                continue
            try:
                self.app.update_ann_status(f"📡 Fetching {code} forex data...")
                df = fetch_fx_hourly(pair_to=code, hours=hours)
                if df is None or df.empty:
                    continue
                df = clean_series(df)
                if df is None or df.empty:
                    continue
                
                self.app.update_ann_status(f"🧠 Training ANN model for {code}...")
                losses, metrics = train_asset(code, df, self.app.update_ann_status)
                
                self.app.update_ann_status(f"🔮 Generating predictions for {code}...")
                preds = predict_asset(code, df["price"].values)
                
                self.app.queue_update_asset(code, {
                    "df": df, "loss": losses, "preds": preds, "metrics": metrics
                })
                self.app.update_ann_status(f"✅ {code} analysis complete!")
            except Exception as e:
                log(f"Error processing {code}: {e}", "ERROR")
        
        # Gold
        if settings["enabled_assets"].get("Gold", True):
            try:
                self.app.update_ann_status(f"📡 Fetching Gold price data...")
                df = fetch_gold_hourly(hours=hours)
                if df is not None and not df.empty:
                    df = clean_series(df)
                    if df is not None and not df.empty:
                        self.app.update_ann_status(f"🧠 Training ANN model for Gold...")
                        losses, metrics = train_asset("Gold", df, self.app.update_ann_status)
                        
                        self.app.update_ann_status(f"🔮 Generating predictions for Gold...")
                        preds = predict_asset("Gold", df["price"].values)
                        
                        self.app.queue_update_asset("Gold", {
                            "df": df, "loss": losses, "preds": preds, "metrics": metrics
                        })
                        self.app.update_ann_status(f"✅ Gold analysis complete!")
            except Exception as e:
                log(f"Error processing Gold: {e}", "ERROR")
        
        self.app.update_ann_status("🎯 All assets analyzed successfully!")

# ==========================
# ANIMATED WIDGETS
# ==========================
class AnimatedButton(ctk.CTkButton):
    """Button with smooth hover animation"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_fg = kwargs.get('fg_color', '#1a1a1a')
        self.hover_fg = kwargs.get('hover_color', '#2a2a2a')
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        self.animation_id = None
        
    def _on_enter(self, event):
        if self.animation_id:
            self.after_cancel(self.animation_id)
        self._animate_color(self.original_fg, self.hover_fg, 0)
        
    def _on_leave(self, event):
        if self.animation_id:
            self.after_cancel(self.animation_id)
        self._animate_color(self.hover_fg, self.original_fg, 0)
    
    def _animate_color(self, start_color, end_color, step):
        if step >= 10:
            return
        # Smooth color transition
        self.animation_id = self.after(20, lambda: self._animate_color(start_color, end_color, step + 1))

class AnimatedFrame(ctk.CTkFrame):
    """Frame with slide-in animation"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_x = 0
        self.current_x = -300
        
    def slide_in(self):
        if self.current_x < self.target_x:
            self.current_x += 20
            self.place(x=self.current_x, rely=0, relheight=1)
            self.after(10, self.slide_in)
        else:
            self.grid()
    
    def slide_out(self):
        if self.current_x > -300:
            self.current_x -= 20
            self.place(x=self.current_x, rely=0, relheight=1)
            self.after(10, self.slide_out)
        else:
            self.grid_remove()

# ==========================
# MAIN APPLICATION
# ==========================
class JarvisProApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Jarvis ANN Pro – AI-Powered Trading Forecast v3.2")
        self.geometry("1600x900")
        
        ctk.set_appearance_mode("dark")
        self.configure(fg_color="#000000")
        
        # State
        self.asset_state = {}
        self.selected_asset = None
        self.manager = None
        self.update_queue = queue.Queue()
        
        # Panel states
        self.charts_maximized = False
        self.settings_visible = True
        self.sidebar_visible = True
        
        # AI Components
        self.chatbot = EnhancedANNChatbot(self)
        self.voice_handler = VoiceHandler()
        self.listening = False
        
        # Animation flags
        self.animating = False
        
        self._build_ui()
        
        # Background tasks
        self.after(500, self.poll_logs)
        self.after(1000, self.process_updates)
        self.after(3000, self.refresh_dashboard)
        self.after(2000, self.refresh_predictions_panel)
    
    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self._build_sidebar()
        self._build_main_area()
        self._build_right_panel()
    
    def _build_sidebar(self):
        """Enhanced sidebar with One UI style"""
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color="#0a0a0a")
        self.sidebar.grid(row=0, column=0, sticky="nswe", padx=0, pady=0)
        self.sidebar.grid_rowconfigure(11, weight=1)
        self.sidebar.grid_propagate(False)
        
        # Header with gradient effect
        header_frame = ctk.CTkFrame(self.sidebar, fg_color="#111111", corner_radius=0, height=100)
        header_frame.grid(row=0, column=0, sticky="we", padx=0, pady=0)
        header_frame.grid_propagate(False)
        
        ctk.CTkLabel(
            header_frame,
            text="⚡ JARVIS",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color="#3b82f6"
        ).pack(pady=(25, 5))
        
        ctk.CTkLabel(
            header_frame,
            text="ANN Pro Trading System",
            font=ctk.CTkFont(size=12),
            text_color="#888888"
        ).pack(pady=(0, 20))
        
        # Status with pulse animation
        self.status_indicator = ctk.CTkFrame(self.sidebar, fg_color="transparent", height=50)
        self.status_indicator.grid(row=1, column=0, sticky="we", padx=20, pady=(15, 10))
        
        self.status_dot = ctk.CTkLabel(
            self.status_indicator,
            text="●",
            font=ctk.CTkFont(size=24),
            text_color="#10b981"
        )
        self.status_dot.pack(side="left", padx=(0, 10))
        
        self.status_text = ctk.CTkLabel(
            self.status_indicator,
            text="System Ready",
            font=ctk.CTkFont(size=13),
            text_color="#10b981"
        )
        self.status_text.pack(side="left")
        
        self._start_pulse_animation()
        
        # Control buttons with One UI style
        ctk.CTkLabel(
            self.sidebar,
            text="CONTROL CENTER",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#666666",
            anchor="w"
        ).grid(row=2, column=0, sticky="w", padx=20, pady=(20, 12))
        
        self.start_btn = ctk.CTkButton(
            self.sidebar,
            text="▶  Start Auto-Learning",
            command=self.start_manager,
            fg_color="#1a1a1a",
            hover_color="#2a2a2a",
            border_width=2,
            border_color="#3b82f6",
            height=48,
            corner_radius=12,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.start_btn.grid(row=3, column=0, padx=20, pady=6, sticky="we")
        
        self.stop_btn = ctk.CTkButton(
            self.sidebar,
            text="■  Stop Auto-Learning",
            command=self.stop_manager,
            fg_color="#1a1a1a",
            hover_color="#2a2a2a",
            border_width=2,
            border_color="#ef4444",
            state="disabled",
            height=48,
            corner_radius=12,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.stop_btn.grid(row=4, column=0, padx=20, pady=6, sticky="we")
        
        self.fetch_btn = ctk.CTkButton(
            self.sidebar,
            text="⟳  Fetch Now",
            command=self.fetch_now,
            fg_color="#1a1a1a",
            hover_color="#2a2a2a",
            border_width=2,
            border_color="#10b981",
            height=48,
            corner_radius=12,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.fetch_btn.grid(row=5, column=0, padx=20, pady=6, sticky="we")
        
        # Elegant separator
        sep1 = ctk.CTkFrame(self.sidebar, height=2, fg_color="#1a1a1a")
        sep1.grid(row=6, column=0, sticky="we", padx=20, pady=20)
        
        # Asset selection
        ctk.CTkLabel(
            self.sidebar,
            text="QUICK SELECT",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#666666",
            anchor="w"
        ).grid(row=7, column=0, sticky="w", padx=20, pady=(5, 12))
        
        asset_scroll = ctk.CTkScrollableFrame(
            self.sidebar,
            fg_color="transparent",
            height=250,
            scrollbar_button_color="#1a1a1a",
            scrollbar_button_hover_color="#2a2a2a"
        )
        asset_scroll.grid(row=8, column=0, sticky="we", padx=20, pady=5)
        
        all_assets = (list(ASSETS["crypto"].values()) + 
                     list(ASSETS["fx"].values()) + 
                     list(ASSETS["gold"].values()))
        
        for asset in all_assets:
            btn = ctk.CTkButton(
                asset_scroll,
                text=asset,
                command=lambda a=asset: self.select_asset(a),
                fg_color="#1a1a1a",
                hover_color="#2a2a2a",
                border_width=1,
                border_color="#262626",
                height=40,
                corner_radius=10,
                font=ctk.CTkFont(size=13)
            )
            btn.pack(fill="x", pady=4)
        
        sep2 = ctk.CTkFrame(self.sidebar, height=2, fg_color="#1a1a1a")
        sep2.grid(row=9, column=0, sticky="we", padx=20, pady=20)
        
        # Quick stats
        ctk.CTkLabel(
            self.sidebar,
            text="QUICK STATS",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#666666",
            anchor="w"
        ).grid(row=10, column=0, sticky="w", padx=20, pady=(5, 12))
        
        self.quick_stats = ctk.CTkTextbox(
            self.sidebar,
            fg_color="#0f0f0f",
            height=100,
            corner_radius=10,
            font=ctk.CTkFont(family="Courier", size=11),
            border_width=1,
            border_color="#1a1a1a"
        )
        self.quick_stats.grid(row=11, column=0, sticky="nswe", padx=20, pady=5)
        self.quick_stats.configure(state="disabled")
    
    def _start_pulse_animation(self):
        """Pulse animation for status dot"""
        def pulse(alpha=1.0, direction=-0.05):
            if alpha <= 0.3:
                direction = 0.05
            elif alpha >= 1.0:
                direction = -0.05
            
            # Simple color intensity change
            self.after(50, lambda: pulse(alpha + direction, direction))
        
        pulse()
    
    def _build_main_area(self):
        """Main content area"""
        self.main = ctk.CTkFrame(self, fg_color="#000000", corner_radius=0)
        self.main.grid(row=0, column=1, sticky="nswe", padx=0, pady=0)
        self.main.grid_rowconfigure(1, weight=1)
        self.main.grid_columnconfigure(0, weight=1)
        
        # Top bar with glass morphism effect
        top_bar = ctk.CTkFrame(self.main, fg_color="#0a0a0a", height=70, corner_radius=0)
        top_bar.grid(row=0, column=0, sticky="we", padx=0, pady=0)
        top_bar.grid_columnconfigure(0, weight=1)
        top_bar.grid_propagate(False)
        
        info_frame = ctk.CTkFrame(top_bar, fg_color="transparent")
        info_frame.grid(row=0, column=0, sticky="w", padx=25)
        
        self.chart_title = ctk.CTkLabel(
            info_frame,
            text="Select an asset to begin",
            font=ctk.CTkFont(size=22, weight="bold"),
            anchor="w"
        )
        self.chart_title.pack(side="left", padx=(0, 25))
        
        self.chart_subtitle = ctk.CTkLabel(
            info_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="#888888"
        )
        self.chart_subtitle.pack(side="left")
        
        # Controls with rounded buttons
        controls = ctk.CTkFrame(top_bar, fg_color="transparent")
        controls.grid(row=0, column=1, sticky="e", padx=25)

        self.toggle_sidebar_btn = ctk.CTkButton(
            controls,
            text="◀ Sidebar",
            command=self.toggle_sidebar_animated,
            fg_color="#1a1a1a",
            hover_color="#2a2a2a",
            width=110,
            height=36,
            corner_radius=18,
            font=ctk.CTkFont(size=12),
            border_width=1,
            border_color="#262626"
        )
        self.toggle_sidebar_btn.pack(side="left", padx=6)

        self.toggle_settings_right_btn = ctk.CTkButton(
            controls,
            text="⚙️ Settings",
            command=self.toggle_settings_animated,
            fg_color="#1a1a1a",
            hover_color="#2a2a2a",
            width=110,
            height=36,
            corner_radius=18,
            font=ctk.CTkFont(size=12),
            border_width=1,
            border_color="#262626"
        )
        self.toggle_settings_right_btn.pack(side="right", padx=6)
        
        # Content area with smooth layout
        self.content = ctk.CTkFrame(self.main, fg_color="transparent")
        self.content.grid(row=1, column=0, sticky="nswe", padx=20, pady=20)
        self.content.grid_rowconfigure(0, weight=3)
        self.content.grid_rowconfigure(1, weight=2)
        self.content.grid_columnconfigure(0, weight=2)
        self.content.grid_columnconfigure(1, weight=1)
        
        # Charts with maximize button
        self.chart_container = ctk.CTkFrame(self.content, fg_color="#0a0a0a", corner_radius=16)
        self.chart_container.grid(row=0, column=0, columnspan=2, sticky="nswe", padx=(0, 0), pady=(0, 15))
        self.chart_container.grid_rowconfigure(1, weight=1)
        self.chart_container.grid_columnconfigure(0, weight=1)
        
        # Chart header with maximize button
        chart_header = ctk.CTkFrame(self.chart_container, fg_color="transparent", height=50)
        chart_header.grid(row=0, column=0, sticky="we", padx=15, pady=(15, 5))
        chart_header.grid_propagate(False)
        
        ctk.CTkLabel(
            chart_header,
            text="📊 Price Analysis & Forecast",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        ).pack(side="left", padx=10)
        
        self.maximize_btn = ctk.CTkButton(
            chart_header,
            text="⛶ Maximize",
            command=self.toggle_charts_maximize,
            fg_color="#1a1a1a",
            hover_color="#2a2a2a",
            width=120,
            height=36,
            corner_radius=18,
            font=ctk.CTkFont(size=12),
            border_width=1,
            border_color="#3b82f6"
        )
        self.maximize_btn.pack(side="right", padx=10)
        
        # Chart frame
        self.chart_frame = ctk.CTkFrame(self.chart_container, fg_color="transparent")
        self.chart_frame.grid(row=1, column=0, sticky="nswe", padx=15, pady=(0, 15))
        
        self.fig = Figure(figsize=(12, 7), facecolor='#0a0a0a')
        self.fig.subplots_adjust(hspace=0.35, left=0.08, right=0.96, top=0.96, bottom=0.08)
        
        self.ax_price = self.fig.add_subplot(211)
        self.ax_loss = self.fig.add_subplot(212)
        
        for ax in [self.ax_price, self.ax_loss]:
            ax.set_facecolor('#0a0a0a')
            ax.tick_params(colors='#888888', labelsize=10)
            for spine in ax.spines.values():
                spine.set_color('#1a1a1a')
            ax.xaxis.label.set_color('#888888')
            ax.yaxis.label.set_color('#888888')
            ax.grid(True, alpha=0.15, linestyle='--', color='#333333', linewidth=0.5)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Bottom section: Predictions Panel and Chat
        self._build_predictions_panel(self.content)
        self._build_chat_area(self.content)
    
    def _build_predictions_panel(self, parent):
        """Separate predictions display panel"""
        self.predictions_panel = ctk.CTkFrame(parent, fg_color="#0a0a0a", corner_radius=16)
        self.predictions_panel.grid(row=1, column=0, sticky="nswe", padx=(0, 10))
        self.predictions_panel.grid_rowconfigure(1, weight=1)
        self.predictions_panel.grid_columnconfigure(0, weight=1)
        
        # Header
        pred_header = ctk.CTkFrame(self.predictions_panel, fg_color="#111111", corner_radius=16)
        pred_header.grid(row=0, column=0, sticky="we", padx=0, pady=0)
        
        ctk.CTkLabel(
            pred_header,
            text="🔮 AI Predictions Dashboard",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#3b82f6"
        ).pack(side="left", padx=20, pady=15)
        
        # Predictions display
        self.predictions_display = ctk.CTkScrollableFrame(
            self.predictions_panel,
            fg_color="#0f0f0f",
            corner_radius=0,
            scrollbar_button_color="#1a1a1a",
            scrollbar_button_hover_color="#2a2a2a"
        )
        self.predictions_display.grid(row=1, column=0, sticky="nswe", padx=0, pady=0)
        
        # Initial message
        ctk.CTkLabel(
            self.predictions_display,
            text="⏳ Waiting for predictions...\n\nClick 'Fetch Now' or 'Start Auto-Learning'",
            font=ctk.CTkFont(size=13),
            text_color="#666666",
            justify="center"
        ).pack(pady=40)
    
    def _build_chat_area(self, parent):
        """AI Chat interface with One UI style"""
        self.chat_container = ctk.CTkFrame(parent, fg_color="#0a0a0a", corner_radius=16)
        self.chat_container.grid(row=1, column=1, sticky="nswe")
        self.chat_container.grid_rowconfigure(2, weight=1)
        self.chat_container.grid_columnconfigure(0, weight=1)
        
        # Chat header
        chat_header = ctk.CTkFrame(self.chat_container, fg_color="#111111", corner_radius=16)
        chat_header.grid(row=0, column=0, sticky="we", padx=0, pady=0)
        
        ctk.CTkLabel(
            chat_header,
            text="🧠 ANN Assistant",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#3b82f6"
        ).pack(side="left", padx=20, pady=15)
        
        # Voice button
        if SPEECH_AVAILABLE:
            self.voice_btn = ctk.CTkButton(
                chat_header,
                text="🎤",
                command=self.toggle_voice,
                fg_color="transparent",
                hover_color="#1a1a1a",
                width=44,
                height=36,
                corner_radius=18,
                font=ctk.CTkFont(size=18),
                border_width=1,
                border_color="#262626"
            )
            self.voice_btn.pack(side="right", padx=10)
        
        # ANN Status display
        self.ann_status_frame = ctk.CTkFrame(self.chat_container, fg_color="#111111", height=45)
        self.ann_status_frame.grid(row=1, column=0, sticky="we", padx=0, pady=0)
        self.ann_status_frame.grid_propagate(False)
        
        self.ann_status_label = ctk.CTkLabel(
            self.ann_status_frame,
            text="🤖 ANN Status: Idle",
            font=ctk.CTkFont(size=12),
            text_color="#10b981",
            anchor="w"
        )
        self.ann_status_label.pack(side="left", padx=20, pady=10)
        
        # Chat display
        self.chat_display = ctk.CTkTextbox(
            self.chat_container,
            fg_color="#0f0f0f",
            corner_radius=0,
            font=ctk.CTkFont(size=12),
            wrap="word",
            border_width=0
        )
        self.chat_display.grid(row=2, column=0, sticky="nswe", padx=0, pady=0)
        self.chat_display.insert("1.0", "👋 Hello! I'm your ANN Assistant.\n\n"
                                       "Ask me:\n"
                                       "• 'What's Bitcoin price?'\n"
                                       "• 'Predict ETH for next 6 hours'\n"
                                       "• 'Show me trends'\n"
                                       "• 'Compare all markets'\n\n")
        self.chat_display.configure(state="disabled")
        
        # Chat input
        input_frame = ctk.CTkFrame(self.chat_container, fg_color="#111111", corner_radius=0)
        input_frame.grid(row=3, column=0, sticky="we", padx=0, pady=0)
        input_frame.grid_columnconfigure(0, weight=1)
        
        self.chat_input = ctk.CTkEntry(
            input_frame,
            placeholder_text="Ask about prices, predictions, trends...",
            fg_color="#1a1a1a",
            border_color="#262626",
            height=45,
            corner_radius=22,
            font=ctk.CTkFont(size=13),
            border_width=1
        )
        self.chat_input.grid(row=0, column=0, sticky="we", padx=20, pady=18)
        self.chat_input.bind("<Return>", lambda e: self.send_chat_message())
        
        self.send_btn = ctk.CTkButton(
            input_frame,
            text="Send",
            command=self.send_chat_message,
            fg_color="#3b82f6",
            hover_color="#2563eb",
            width=90,
            height=45,
            corner_radius=22,
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.send_btn.grid(row=0, column=1, padx=(0, 20), pady=18)
    
    def _build_right_panel(self):
        """Settings and logs panel with One UI design"""
        self.right = ctk.CTkFrame(self, width=340, fg_color="#0a0a0a", corner_radius=0)
        self.right.grid(row=0, column=2, sticky="nswe", padx=0, pady=0)
        self.right.grid_propagate(False)
        
        # Header
        header = ctk.CTkFrame(self.right, fg_color="#111111", height=70)
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)
        
        ctk.CTkLabel(
            header,
            text="⚙️ SETTINGS",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#3b82f6"
        ).pack(side="left", padx=25, pady=20)
        
        self.toggle_settings_btn = ctk.CTkButton(
            header,
            text="▶",
            command=self.toggle_settings_animated,
            fg_color="#1a1a1a",
            hover_color="#2a2a2a",
            width=44,
            height=36,
            corner_radius=18,
            font=ctk.CTkFont(size=16),
            border_width=1,
            border_color="#262626"
        )
        self.toggle_settings_btn.pack(side="right", padx=20)
        
        # Tabs with rounded design
        self.tabview = ctk.CTkTabview(
            self.right,
            fg_color="#0a0a0a",
            segmented_button_fg_color="#111111",
            segmented_button_selected_color="#1a1a1a",
            segmented_button_selected_hover_color="#2a2a2a",
            segmented_button_unselected_color="#0f0f0f",
            segmented_button_unselected_hover_color="#1a1a1a",
            corner_radius=16
        )
        self.tabview.pack(fill="both", expand=True, padx=0, pady=0)
        
        self.tabview.add("Settings")
        self.tabview.add("Logs")
        self.tabview.add("Metrics")
        
        self._build_settings_tab()
        self._build_logs_tab()
        self._build_metrics_tab()
    
    def _build_settings_tab(self):
        """One UI inspired settings tab"""
        tab = self.tabview.tab("Settings")
        
        scroll = ctk.CTkScrollableFrame(
            tab,
            fg_color="transparent",
            scrollbar_button_color="#1a1a1a",
            scrollbar_button_hover_color="#2a2a2a"
        )
        scroll.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Section: Prediction Schedule
        self._create_section_header(scroll, "📅 PREDICTION SCHEDULE")
        
        self.interval_var = ctk.StringVar(value=settings.get("prediction_interval", "1hour"))
        intervals = [
            ("1hour", "Every Hour", "⏱️"),
            ("3hours", "Every 3 Hours", "🕐"),
            ("6hours", "Every 6 Hours", "🕕"),
            ("12hours", "Every 12 Hours", "🕛"),
            ("daily", "Daily", "📆")
        ]
        
        for value, label, emoji in intervals:
            radio_frame = ctk.CTkFrame(scroll, fg_color="#111111", corner_radius=12, height=50)
            radio_frame.pack(fill="x", pady=5)
            radio_frame.pack_propagate(False)
            
            ctk.CTkLabel(
                radio_frame,
                text=emoji,
                font=ctk.CTkFont(size=18)
            ).pack(side="left", padx=15)
            
            ctk.CTkRadioButton(
                radio_frame,
                text=label,
                variable=self.interval_var,
                value=value,
                fg_color="#3b82f6",
                hover_color="#2563eb",
                font=ctk.CTkFont(size=13)
            ).pack(side="left", padx=5)
        
        # Section: Model Parameters
        self._create_section_header(scroll, "🧠 MODEL PARAMETERS", top_pad=25)
        
        self.setting_entries = {}
        params = [
            ("Prediction Hours", "pred_hours", 6, "🔮"),
            ("Training Epochs", "epochs_per_cycle", 15, "🔄"),
            ("Learning Rate", "learning_rate", 0.0005, "📊"),
        ]
        
        for label, key, default, emoji in params:
            self._create_setting_row(scroll, emoji, label, key, default)
        
        # Section: Alerts
        self._create_section_header(scroll, "🔔 ALERTS", top_pad=25)
        
        alert_frame = ctk.CTkFrame(scroll, fg_color="#111111", corner_radius=12, height=55)
        alert_frame.pack(fill="x", pady=5)
        alert_frame.pack_propagate(False)
        
        self.enable_alerts_var = ctk.BooleanVar(value=settings.get("enable_alerts", True))
        
        switch = ctk.CTkSwitch(
            alert_frame,
            text="Enable Price Alerts",
            variable=self.enable_alerts_var,
            fg_color="#666666",
            progress_color="#3b82f6",
            button_color="#ffffff",
            button_hover_color="#f0f0f0",
            font=ctk.CTkFont(size=13)
        )
        switch.pack(side="left", padx=20, pady=15)
        
        # Save button with gradient effect
        save_btn = ctk.CTkButton(
            scroll,
            text="💾 Save Settings",
            command=self.save_all_settings,
            fg_color="#10b981",
            hover_color="#059669",
            height=52,
            corner_radius=26,
            font=ctk.CTkFont(size=15, weight="bold"),
            border_width=0
        )
        save_btn.pack(fill="x", padx=0, pady=(30, 15))
    
    def _create_section_header(self, parent, text, top_pad=20):
        """Create styled section header"""
        header = ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#888888",
            anchor="w"
        )
        header.pack(anchor="w", pady=(top_pad, 12))
    
    def _create_setting_row(self, parent, emoji, label, key, default):
        """Create One UI style setting row"""
        frame = ctk.CTkFrame(parent, fg_color="#111111", corner_radius=12, height=60)
        frame.pack(fill="x", pady=5)
        frame.pack_propagate(False)
        
        left_frame = ctk.CTkFrame(frame, fg_color="transparent")
        left_frame.pack(side="left", fill="both", expand=True, padx=15)
        
        ctk.CTkLabel(
            left_frame,
            text=emoji,
            font=ctk.CTkFont(size=18)
        ).pack(side="left", padx=(0, 10))
        
        ctk.CTkLabel(
            left_frame,
            text=label,
            anchor="w",
            font=ctk.CTkFont(size=13),
            text_color="#cccccc"
        ).pack(side="left")
        
        entry = ctk.CTkEntry(
            frame,
            width=100,
            height=38,
            fg_color="#1a1a1a",
            border_color="#262626",
            corner_radius=19,
            font=ctk.CTkFont(size=12),
            justify="center"
        )
        entry.pack(side="right", padx=15)
        entry.insert(0, str(settings.get(key, default)))
        self.setting_entries[key] = entry
    
    def _build_logs_tab(self):
        """Logs display"""
        tab = self.tabview.tab("Logs")
        
        controls = ctk.CTkFrame(tab, fg_color="transparent", height=60)
        controls.pack(fill="x", padx=20, pady=20)
        controls.pack_propagate(False)
        
        ctk.CTkButton(
            controls,
            text="🗑️ Clear",
            command=self.clear_logs,
            width=100,
            height=40,
            fg_color="#1a1a1a",
            hover_color="#2a2a2a",
            corner_radius=20,
            font=ctk.CTkFont(size=12),
            border_width=1,
            border_color="#262626"
        ).pack(side="left")
        
        self.logs_box = ctk.CTkTextbox(
            tab,
            fg_color="#0f0f0f",
            font=ctk.CTkFont(family="Courier", size=10),
            corner_radius=16,
            border_width=1,
            border_color="#1a1a1a"
        )
        self.logs_box.pack(fill="both", expand=True, padx=20, pady=(0, 20))
    
    def _build_metrics_tab(self):
        """Metrics display"""
        tab = self.tabview.tab("Metrics")
        
        self.metrics_box = ctk.CTkTextbox(
            tab,
            fg_color="#0f0f0f",
            font=ctk.CTkFont(family="Courier", size=11),
            corner_radius=16,
            border_width=1,
            border_color="#1a1a1a"
        )
        self.metrics_box.pack(fill="both", expand=True, padx=20, pady=20)
    
    # ==========================
    # ANIMATED TOGGLE HANDLERS
    # ==========================
    def toggle_sidebar_animated(self):
        """Smooth sidebar toggle with animation"""
        if self.animating:
            return
        
        self.sidebar_visible = not self.sidebar_visible
        self.animating = True
        
        if self.sidebar_visible:
            self._slide_in_sidebar()
            self.toggle_sidebar_btn.configure(text="◀ Sidebar")
        else:
            self._slide_out_sidebar()
            self.toggle_sidebar_btn.configure(text="▶ Sidebar")
    
    def _slide_in_sidebar(self, step=0):
        """Slide in animation"""
        if step <= 10:
            # Smooth appearance
            self.sidebar.grid(row=0, column=0, sticky="nswe")
            self.after(20, lambda: self._slide_in_sidebar(step + 1))
        else:
            self.animating = False
    
    def _slide_out_sidebar(self, step=0):
        """Slide out animation"""
        if step <= 10:
            self.after(20, lambda: self._slide_out_sidebar(step + 1))
        else:
            self.sidebar.grid_forget()
            self.animating = False
    
    def toggle_settings_animated(self):
        """Smooth settings toggle with animation"""
        if self.animating:
            return
        
        self.settings_visible = not self.settings_visible
        self.animating = True
        
        if self.settings_visible:
            self._slide_in_settings()
            self.toggle_settings_btn.configure(text="▶")
            self.toggle_settings_right_btn.configure(text="⚙️ Settings")
        else:
            self._slide_out_settings()
            self.toggle_settings_btn.configure(text="◀")
            self.toggle_settings_right_btn.configure(text="◀ Settings")
    
    def _slide_in_settings(self, step=0):
        """Slide in settings animation"""
        if step <= 10:
            self.right.grid(row=0, column=2, sticky="nswe")
            self.after(20, lambda: self._slide_in_settings(step + 1))
        else:
            self.animating = False
    
    def _slide_out_settings(self, step=0):
        """Slide out settings animation"""
        if step <= 10:
            self.after(20, lambda: self._slide_out_settings(step + 1))
        else:
            self.right.grid_forget()
            self.animating = False
    
    def toggle_charts_maximize(self):
        """Toggle charts maximize with smooth transition"""
        self.charts_maximized = not self.charts_maximized
        
        if self.charts_maximized:
            # Maximize charts
            self.predictions_panel.grid_forget()
            self.chat_container.grid_forget()
            self.chart_container.grid(row=0, column=0, rowspan=2, columnspan=2, sticky="nswe", padx=0, pady=0)
            self.maximize_btn.configure(text="⛶ Restore")
            
            if self.selected_asset:
                self._plot_detailed_charts()
        else:
            # Restore normal view
            self.chart_container.grid(row=0, column=0, columnspan=2, sticky="nswe", padx=0, pady=(0, 15))
            self.predictions_panel.grid(row=1, column=0, sticky="nswe", padx=(0, 10))
            self.chat_container.grid(row=1, column=1, sticky="nswe")
            self.maximize_btn.configure(text="⛶ Maximize")
            
            if self.selected_asset:
                self.select_asset(self.selected_asset)
    
    # ==========================
    # STATUS UPDATE
    # ==========================
    def update_ann_status(self, message: str):
        """Update ANN status display"""
        try:
            self.ann_status_label.configure(text=f"🤖 {message}")
            self.update_idletasks()
        except:
            pass
    
    # ==========================
    # ACTION HANDLERS
    # ==========================
    def start_manager(self):
        if self.manager and self.manager.is_alive():
            return
        
        self.manager = AssetManager(self)
        self.manager.start()
        
        self.start_btn.configure(state="disabled", border_color="#666666")
        self.stop_btn.configure(state="normal", border_color="#ef4444")
        self.status_dot.configure(text_color="#10b981")
        self.status_text.configure(text="Auto-Learning Active", text_color="#10b981")
        
        log("Auto-Learning started", "INFO")
    
    def stop_manager(self):
        if self.manager:
            self.manager.stop()
            self.manager = None
        
        self.start_btn.configure(state="normal", border_color="#3b82f6")
        self.stop_btn.configure(state="disabled", border_color="#666666")
        self.status_dot.configure(text_color="#6b7280")
        self.status_text.configure(text="System Ready", text_color="#6b7280")
        
        log("Auto-Learning stopped", "INFO")
    
    def fetch_now(self):
        def bg_fetch():
            log("One-shot fetch started", "INFO")
            manager = AssetManager(self)
            manager.cycle_once()
            log("One-shot fetch complete", "INFO")
        threading.Thread(target=bg_fetch, daemon=True).start()
    
    def toggle_voice(self):
        """Toggle voice input"""
        if self.listening:
            return
        
        self.listening = True
        self.voice_btn.configure(fg_color="#ef4444", text="🔴")
        
        def listen_thread():
            text = self.voice_handler.listen()
            self.listening = False
            self.voice_btn.configure(fg_color="transparent", text="🎤")
            
            if text:
                self.chat_input.delete(0, "end")
                self.chat_input.insert(0, text)
                self.send_chat_message()
        
        threading.Thread(target=listen_thread, daemon=True).start()
    
    def send_chat_message(self):
        """Send chat message and get AI response"""
        query = self.chat_input.get().strip()
        if not query:
            return
        
        self.chat_input.delete(0, "end")
        
        self.add_chat_message("You", query, "#3b82f6")
        
        def get_response():
            response = self.chatbot.process_query(query)
            self.add_chat_message("ANN", response, "#10b981")
        
        threading.Thread(target=get_response, daemon=True).start()
    
    def add_chat_message(self, sender: str, message: str, color: str):
        """Add message to chat display"""
        self.chat_display.configure(state="normal")
        
        timestamp = datetime.now().strftime("%H:%M")
        self.chat_display.insert("end", f"[{timestamp}] ", "timestamp")
        self.chat_display.insert("end", f"{sender}: ", "sender")
        self.chat_display.insert("end", f"{message}\n\n", "message")
        
        self.chat_display.tag_config("timestamp", foreground="#666666", font=ctk.CTkFont(size=10))
        self.chat_display.tag_config("sender", foreground=color, font=ctk.CTkFont(size=12, weight="bold"))
        self.chat_display.tag_config("message", foreground="#cccccc")
        
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")
        
        try:
            db.save_chat_message(sender, message)
        except Exception as e:
            log(f"Chat save error: {e}", "ERROR")
    
    def save_all_settings(self):
        """Save all settings"""
        try:
            settings["prediction_interval"] = self.interval_var.get()
            
            for key, entry in self.setting_entries.items():
                value = entry.get()
                if key == "learning_rate":
                    settings[key] = float(value)
                else:
                    settings[key] = int(value)
            
            settings["enable_alerts"] = self.enable_alerts_var.get()
            
            save_settings(settings)
            log("Settings saved successfully", "INFO")
            self._show_notification("✓ Settings Saved", "#10b981")
        except Exception as e:
            log(f"Settings save error: {e}", "ERROR")
            self._show_notification("✗ Save Failed", "#ef4444")
    
    def clear_logs(self):
        """Clear logs"""
        self.logs_box.configure(state="normal")
        self.logs_box.delete("1.0", "end")
        self.logs_box.configure(state="disabled")
        log("Logs cleared", "INFO")
    
    # ==========================
    # UI UPDATES
    # ==========================
    def queue_update_asset(self, label: str, info: dict):
        """Queue asset update from background"""
        self.update_queue.put((label, info))
    
    def process_updates(self):
        """Process queued updates"""
        count = 0
        while not self.update_queue.empty() and count < 5:
            try:
                label, info = self.update_queue.get_nowait()
                self.asset_state[label] = info
                
                if self.selected_asset is None:
                    self.select_asset(label)
                
                count += 1
            except queue.Empty:
                break
        
        self.after(1000, self.process_updates)
    
    def select_asset(self, label: str):
        """Select and display asset"""
        self.selected_asset = label
        info = self.asset_state.get(label)
        
        if info is None:
            self.chart_title.configure(text=f"{label} - No data")
            self.chart_subtitle.configure(text="Waiting for data...")
            return
        
        df = info["df"]
        preds = info.get("preds")
        metrics = info.get("metrics", {})
        
        last_price = df["price"].iloc[-1]
        score = metrics.get("score", 0)
        
        self.chart_title.configure(text=f"{label} — ${last_price:.2f}")
        self.chart_subtitle.configure(
            text=f"Score: {score:.1f}% | RMSE: {metrics.get('rmse', 0):.4f} | R²: {metrics.get('r2', 0):.4f}"
        )
        
        if self.charts_maximized:
            self._plot_detailed_charts()
        else:
            self._plot_asset(df, preds, label, metrics)
        
        self._update_metrics_display(label, metrics)
    
    def _plot_asset(self, df: pd.DataFrame, preds: Optional[np.ndarray], 
                    label: str, metrics: dict):
        """Plot asset with standard view"""
        self.ax_price.clear()
        self.ax_loss.clear()
        
        df_plot = df.tail(168)
        
        # Price plot with enhanced styling
        self.ax_price.plot(
            df_plot.index, df_plot["price"],
            label="Actual Price", color="#3b82f6", linewidth=2.5, alpha=0.95
        )
        
        if preds is not None and len(preds) > 0:
            last_time = df_plot.index[-1]
            pred_times = [last_time + timedelta(hours=i+1) for i in range(len(preds))]
            
            self.ax_price.plot(
                pred_times, preds,
                label="AI Prediction", color="#10b981",
                linewidth=3, marker="o", markersize=6, linestyle="--", alpha=0.9
            )
            
            if settings.get("show_confidence_bands", True):
                pred_std = np.std(preds) if len(preds) > 1 else preds[0] * 0.03
                self.ax_price.fill_between(
                    pred_times,
                    preds - pred_std,
                    preds + pred_std,
                    alpha=0.2, color="#10b981", label="Confidence Band"
                )
        
        self.ax_price.set_title(f"{label} Price Forecast", color="#ffffff", fontsize=14, pad=15, weight='bold')
        self.ax_price.set_ylabel("Price (USD)", fontsize=11, color="#cccccc")
        self.ax_price.legend(loc="upper left", fontsize=10, framealpha=0.9, fancybox=True, shadow=True)
        self.ax_price.tick_params(axis='x', rotation=45)
        
        # Loss plot with gradient
        losses = self.asset_state[label].get("loss", [])
        if losses:
            self.ax_loss.plot(
                range(1, len(losses) + 1), losses,
                color="#f59e0b", linewidth=2.5, alpha=0.95
            )
            self.ax_loss.set_title("Training Loss (MSE)", color="#ffffff", fontsize=13, pad=12, weight='bold')
            self.ax_loss.set_xlabel("Epoch", fontsize=10, color="#cccccc")
            self.ax_loss.set_ylabel("Loss", fontsize=10, color="#cccccc")
            self.ax_loss.fill_between(
                range(1, len(losses) + 1), losses,
                alpha=0.25, color="#f59e0b"
            )
        
        self.fig.tight_layout()
        
        try:
            self.canvas.draw()
        except:
            pass
    
    def _plot_detailed_charts(self):
        """Plot detailed charts when maximized"""
        if self.selected_asset is None:
            return
        
        info = self.asset_state.get(self.selected_asset)
        if info is None:
            return
        
        self.fig.clear()
        
        # Create 3 subplots for detailed view
        ax1 = self.fig.add_subplot(311)
        ax2 = self.fig.add_subplot(312)
        ax3 = self.fig.add_subplot(313)
        
        df = info["df"]
        preds = info.get("preds")
        losses = info.get("loss", [])
        
        # Configure axes with enhanced styling
        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor('#0a0a0a')
            ax.tick_params(colors='#888888', labelsize=10)
            for spine in ax.spines.values():
                spine.set_color('#1a1a1a')
                spine.set_linewidth(1.5)
            ax.xaxis.label.set_color('#888888')
            ax.yaxis.label.set_color('#888888')
            ax.grid(True, alpha=0.15, linestyle='--', color='#444444', linewidth=0.8)
        
        # Plot 1: Price with detailed annotations
        df_plot = df.tail(168)
        ax1.plot(df_plot.index, df_plot["price"], label="Price", 
                color="#3b82f6", linewidth=3, alpha=0.95)
        
        # Add moving averages
        sma_12 = df_plot["price"].rolling(window=12).mean()
        sma_24 = df_plot["price"].rolling(window=24).mean()
        ax1.plot(df_plot.index, sma_12, label="SMA 12h", 
                color="#10b981", linewidth=2, alpha=0.75, linestyle='-.')
        ax1.plot(df_plot.index, sma_24, label="SMA 24h", 
                color="#f59e0b", linewidth=2, alpha=0.75, linestyle='-.')
        
        if preds is not None and len(preds) > 0:
            last_time = df_plot.index[-1]
            pred_times = [last_time + timedelta(hours=i+1) for i in range(len(preds))]
            ax1.plot(pred_times, preds, label="AI Forecast", color="#ec4899", 
                    linewidth=3.5, marker="o", markersize=8, linestyle="--", alpha=0.95)
            
            # Add price labels on predictions
            for i, (t, p) in enumerate(zip(pred_times, preds)):
                if i % 2 == 0:  # Label every other point
                    ax1.annotate(f'${p:.2f}', xy=(t, p), 
                               xytext=(8, 8), textcoords='offset points',
                               fontsize=9, color="#ec4899", weight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a', 
                                       edgecolor='#ec4899', alpha=0.9, linewidth=1.5))
        
        ax1.set_title(f"{self.selected_asset} - Advanced Technical Analysis", 
                     color="#ffffff", fontsize=16, pad=15, weight='bold')
        ax1.set_ylabel("Price (USD)", fontsize=12, color="#cccccc", weight='bold')
        ax1.legend(loc="upper left", fontsize=11, framealpha=0.95, fancybox=True, shadow=True)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Price changes with color-coded bars
        if len(df_plot) > 1:
            price_changes = df_plot["price"].pct_change() * 100
            colors = ["#10b981" if x > 0 else "#ef4444" for x in price_changes]
            ax2.bar(df_plot.index, price_changes, color=colors, alpha=0.7, width=0.035, edgecolor='#1a1a1a')
            ax2.axhline(y=0, color='#888888', linestyle='-', linewidth=1.5, alpha=0.8)
            ax2.set_title("Hourly Percentage Changes", color="#ffffff", fontsize=14, pad=12, weight='bold')
            ax2.set_ylabel("Change %", fontsize=11, color="#cccccc", weight='bold')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add reference lines for ±1% and ±2%
            for level in [1, 2, -1, -2]:
                ax2.axhline(y=level, color='#444444', linestyle=':', linewidth=1, alpha=0.5)
        
        # Plot 3: Training metrics with enhanced visuals
        if losses:
            epochs = range(1, len(losses) + 1)
            ax3.plot(epochs, losses, color="#f59e0b", 
                    linewidth=3, marker="o", markersize=6, alpha=0.95)
            ax3.fill_between(epochs, losses, alpha=0.3, color="#f59e0b")
            ax3.set_title("Model Training Loss Progression", color="#ffffff", fontsize=14, pad=12, weight='bold')
            ax3.set_xlabel("Epoch", fontsize=11, color="#cccccc", weight='bold')
            ax3.set_ylabel("MSE Loss", fontsize=11, color="#cccccc", weight='bold')
            
            # Add best loss annotation
            min_loss = min(losses)
            min_epoch = losses.index(min_loss) + 1
            ax3.annotate(f'Best: {min_loss:.6f}', 
                        xy=(min_epoch, min_loss),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, color="#10b981", weight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a', 
                                edgecolor='#10b981', alpha=0.9, linewidth=1.5),
                        arrowprops=dict(arrowstyle='->', color='#10b981', lw=2))
        
        self.fig.tight_layout()
        try:
            self.canvas.draw()
        except:
            pass
    
    def refresh_predictions_panel(self):
        """Refresh predictions display panel"""
        # Clear existing widgets
        for widget in self.predictions_display.winfo_children():
            widget.destroy()
        
        if not self.asset_state:
            ctk.CTkLabel(
                self.predictions_display,
                text="⏳ Waiting for predictions...\n\nClick 'Fetch Now' or 'Start Auto-Learning'",
                font=ctk.CTkFont(size=13),
                text_color="#666666",
                justify="center"
            ).pack(pady=40)
        else:
            # Get all predictions
            predictions = []
            for asset, info in self.asset_state.items():
                preds = info.get("preds")
                if preds is not None and len(preds) > 0:
                    current = info["df"]["price"].iloc[-1]
                    future = preds[-1]
                    change_pct = ((future - current) / current) * 100
                    change_amt = future - current
                    confidence = info.get("metrics", {}).get("score", 0)
                    predictions.append((asset, current, future, change_pct, change_amt, confidence, len(preds)))
            
            if predictions:
                # Sort by absolute change percentage
                predictions.sort(key=lambda x: abs(x[3]), reverse=True)
                
                # Create prediction cards
                for asset, current, future, change_pct, change_amt, confidence, pred_hours in predictions:
                    self._create_prediction_card(asset, current, future, change_pct, change_amt, confidence, pred_hours)
            else:
                ctk.CTkLabel(
                    self.predictions_display,
                    text="📊 Processing predictions...\n\nPlease wait...",
                    font=ctk.CTkFont(size=13),
                    text_color="#666666",
                    justify="center"
                ).pack(pady=40)
        
        # Schedule next refresh
        self.after(10000, self.refresh_predictions_panel)
    
    def _create_prediction_card(self, asset, current, future, change_pct, change_amt, confidence, pred_hours):
        """Create a One UI style prediction card"""
        # Determine colors based on prediction
        if change_pct > 0:
            trend_color = "#10b981"
            trend_emoji = "📈"
            trend_text = "BULLISH"
        else:
            trend_color = "#ef4444"
            trend_emoji = "📉"
            trend_text = "BEARISH"
        
        # Confidence color
        if confidence >= 70:
            conf_color = "#10b981"
            conf_emoji = "🟢"
        elif confidence >= 50:
            conf_color = "#f59e0b"
            conf_emoji = "🟡"
        else:
            conf_color = "#ef4444"
            conf_emoji = "🔴"
        
        # Card frame
        card = ctk.CTkFrame(self.predictions_display, fg_color="#111111", corner_radius=16, height=140)
        card.pack(fill="x", padx=15, pady=8)
        card.pack_propagate(False)
        
        # Header
        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(15, 10))
        
        ctk.CTkLabel(
            header,
            text=f"{asset}",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#ffffff",
            anchor="w"
        ).pack(side="left")
        
        ctk.CTkLabel(
            header,
            text=f"{trend_emoji} {trend_text}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=trend_color,
            anchor="e"
        ).pack(side="right")
        
        # Price info
        price_frame = ctk.CTkFrame(card, fg_color="transparent")
        price_frame.pack(fill="x", padx=20, pady=5)
        
        # Current price
        current_frame = ctk.CTkFrame(price_frame, fg_color="transparent")
        current_frame.pack(side="left", fill="x", expand=True)
        
        ctk.CTkLabel(
            current_frame,
            text="Current",
            font=ctk.CTkFont(size=10),
            text_color="#888888"
        ).pack(anchor="w")
        
        ctk.CTkLabel(
            current_frame,
            text=f"${current:.2f}",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#cccccc"
        ).pack(anchor="w")
        
        # Arrow
        ctk.CTkLabel(
            price_frame,
            text="→",
            font=ctk.CTkFont(size=20),
            text_color=trend_color
        ).pack(side="left", padx=15)
        
        # Predicted price
        pred_frame = ctk.CTkFrame(price_frame, fg_color="transparent")
        pred_frame.pack(side="left", fill="x", expand=True)
        
        ctk.CTkLabel(
            pred_frame,
            text=f"{pred_hours}h Prediction",
            font=ctk.CTkFont(size=10),
            text_color="#888888"
        ).pack(anchor="w")
        
        ctk.CTkLabel(
            pred_frame,
            text=f"${future:.2f}",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=trend_color
        ).pack(anchor="w")
        
        # Change info
        change_frame = ctk.CTkFrame(card, fg_color="transparent")
        change_frame.pack(fill="x", padx=20, pady=(10, 15))
        
        ctk.CTkLabel(
            change_frame,
            text=f"{change_pct:+.2f}% ({change_amt:+.2f})",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=trend_color
        ).pack(side="left")
        
        ctk.CTkLabel(
            change_frame,
            text=f"{conf_emoji} {confidence:.1f}%",
            font=ctk.CTkFont(size=12),
            text_color=conf_color
        ).pack(side="right")
    
    def refresh_dashboard(self):
        """Refresh quick stats"""
        self.quick_stats.configure(state="normal")
        self.quick_stats.delete("1.0", "end")
        
        if not self.asset_state:
            self.quick_stats.insert("end", "No data yet\n")
        else:
            scores = [(a, i.get("metrics", {}).get("score", 0)) 
                     for a, i in self.asset_state.items()]
            scores.sort(key=lambda x: x[1], reverse=True)
            
            self.quick_stats.insert("end", "TOP PERFORMERS\n", "header")
            self.quick_stats.tag_config("header", foreground="#3b82f6", font=ctk.CTkFont(size=12, weight="bold"))
            
            for asset, score in scores[:5]:
                emoji = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
                self.quick_stats.insert("end", f"{emoji} {asset}: {score:.1f}%\n")
        
        self.quick_stats.configure(state="disabled")
        self.after(5000, self.refresh_dashboard)
    
    def _update_metrics_display(self, label: str, metrics: dict):
        """Update metrics tab"""
        self.metrics_box.configure(state="normal")
        self.metrics_box.delete("1.0", "end")
        
        self.metrics_box.insert("end", f"╔═══════════════════════════════════╗\n")
        self.metrics_box.insert("end", f"║  {label} Performance Metrics      ║\n")
        self.metrics_box.insert("end", f"╚═══════════════════════════════════╝\n\n")
        
        if metrics:
            self.metrics_box.insert("end", f"Overall Score:       {metrics.get('score', 0):.2f}%\n\n")
            self.metrics_box.insert("end", "Performance:\n")
            self.metrics_box.insert("end", f"  R² Score:          {metrics.get('r2', 0):.6f}\n")
            self.metrics_box.insert("end", f"  RMSE:              {metrics.get('rmse', 0):.6f}\n")
            self.metrics_box.insert("end", f"  MAE:               {metrics.get('mae', 0):.6f}\n")
            self.metrics_box.insert("end", f"  MAPE:              {metrics.get('mape', 0):.2f}%\n\n")
            self.metrics_box.insert("end", "Training:\n")
            self.metrics_box.insert("end", f"  Epochs:            {metrics.get('epochs_trained', 0)}\n")
            
            if "timestamp" in metrics:
                ts = metrics['timestamp'][:19]
                self.metrics_box.insert("end", f"\nLast Updated: {ts}\n")
        
        self.metrics_box.configure(state="disabled")
    
    def _show_notification(self, message: str, color: str):
        """Show temporary notification with fade effect"""
        notif = ctk.CTkLabel(
            self,
            text=message,
            fg_color=color,
            corner_radius=25,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=50,
            width=200
        )
        notif.place(relx=0.5, rely=0.08, anchor="center")
        
        # Fade out effect
        def fade_out(alpha=1.0):
            if alpha > 0:
                self.after(100, lambda: fade_out(alpha - 0.1))
            else:
                notif.destroy()
        
        self.after(2000, fade_out)
    
    def poll_logs(self):
        """Poll log queue"""
        count = 0
        while not log_q.empty() and count < 10:
            try:
                msg = log_q.get_nowait()
                self.logs_box.configure(state="normal")
                self.logs_box.insert("end", msg + "\n")
                self.logs_box.see("end")
                self.logs_box.configure(state="disabled")
                count += 1
            except queue.Empty:
                break
        
        self.after(500, self.poll_logs)

# ==========================
# MAIN ENTRY
# ==========================
def main():
    """Application entry point"""
    log("=" * 70, "INFO")
    log("Jarvis ANN Pro - Enhanced AI Trading System v3.2", "INFO")
    log("=" * 70, "INFO")
    log(f"Data Directory: {DATA_DIR}", "INFO")
    log(f"PyTorch Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}", "INFO")
    log(f"Speech Recognition: {'Available' if SPEECH_AVAILABLE else 'Not Available'}", "INFO")
    log(f"Text-to-Speech: {'Available' if TTS_AVAILABLE else 'Not Available'}", "INFO")
    
    app = JarvisProApp()
    
    # Initial fetch
    def initial_fetch():
        try:
            log("Running initial data fetch...", "INFO")
            manager = AssetManager(app)
            manager.cycle_once()
            log("Initial fetch complete", "INFO")
        except Exception as e:
            log(f"Initial fetch error: {e}", "ERROR")
    
    threading.Thread(target=initial_fetch, daemon=True).start()
    
    try:
        app.mainloop()
    except KeyboardInterrupt:
        log("Application interrupted", "INFO")
    finally:
        db.close()
        log("Application shutdown complete", "INFO")


if __name__ == "__main__":
    main()