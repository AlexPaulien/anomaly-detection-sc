import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from src.config import DATA_DIR, MODELS_DIR

import json

WINDOW_L = 13
SEASONAL_PERIOD = 52


def robust_scale(sales):
    """Per-series robust normalization: (sales - median) / IQR.

    GLOBAL (non-adaptive) on purpose. A late level shift barely moves the series
    median, so the shifted weeks show as a sustained offset — the AE keeps
    flagging them. This is exactly what the IF's adaptive 13-week rolling window
    cannot do (it absorbs the shift). The non-adaptive scaler is *why* the AE
    beats the IF on sustained regime change.
    """
    med = float(np.median(sales))
    q1, q3 = np.percentile(sales, [25, 75])
    iqr = q3 - q1
    if iqr <= 1e-6:                      # near-constant series: fall back to std, then 1
        iqr = float(np.std(sales)) if np.std(sales) > 1e-6 else 1.0
    return (sales - med) / iqr, med, float(iqr)


def build_series_channels(s):
    """One series (sorted by wm_yr_wk) -> (T, 3) array [sales_norm, sin, cos]."""
    sales = s["sales"].to_numpy(dtype=float)
    normed, med, iqr = robust_scale(sales)
    woy = s["wm_yr_wk"].to_numpy() % 100
    ang = 2 * np.pi * (woy % SEASONAL_PERIOD) / SEASONAL_PERIOD
    chans = np.stack([normed, np.sin(ang), np.cos(ang)], axis=1)
    return chans, med, iqr


def make_windows(chans, L=WINDOW_L):
    """Sliding windows over one series. Returns (n, L, 3) and center indices."""
    T = len(chans)
    if T < L:
        return np.empty((0, L, 3), dtype=float), np.array([], dtype=int)
    starts = np.arange(T - L + 1)
    wins = np.stack([chans[i:i + L] for i in starts])
    centers = starts + L // 2            # a window scores its CENTER week
    return wins, centers


def build_training_windows(sample_per_series=50, max_windows=800_000, seed=42):
    """All dense series -> training windows (N, L, 3), subsampled for local training.

    The AE windows over ALL active weeks (it needs no YoY), so it can even score
    the early-life weeks the IF cannot — a minor AE advantage.
    """
    feats = pd.read_parquet(DATA_DIR / "features.parquet")
    rng = np.random.default_rng(seed)
    parts = []
    for _, s in feats.sort_values("wm_yr_wk").groupby("id"):
        chans, _, _ = build_series_channels(s)
        wins, _ = make_windows(chans)
        if len(wins) == 0:
            continue
        if sample_per_series and len(wins) > sample_per_series:
            wins = wins[rng.choice(len(wins), sample_per_series, replace=False)]
        parts.append(wins)
    X = np.concatenate(parts).astype(np.float32)
    if len(X) > max_windows:
        X = X[rng.choice(len(X), max_windows, replace=False)]
    return X


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LSTMAutoencoder(nn.Module):
    """Sequence autoencoder: encode an L-step window to a small latent, decode it
    back. Trained to reconstruct NORMAL windows; a window whose shape it hasn't
    learned (a level shift, a shock) reconstructs poorly -> high error.
    """
    def __init__(self, n_features=3, hidden=32, latent=16, seq_len=WINDOW_L):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):                        # x: (B, L, n_features)
        _, (h, _) = self.encoder(x)              # h: (1, B, hidden)
        z = self.to_latent(h[-1])                # (B, latent)
        dec_in = self.from_latent(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder(dec_in)        # (B, L, hidden)
        return self.out(dec_out)                 # (B, L, n_features)


def reconstruction_error(model, X, device, batch=4096, channel=0):
    """Per-window reconstruction error, measured on ONE channel (sales by default).

    The seasonal sin/cos channels are deterministic and trivially reconstructed,
    so scoring on the sales channel alone keeps the anomaly signal undiluted.
    """
    import numpy as np
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], device=device)
            rb = model(xb)
            e = ((rb[:, :, channel] - xb[:, :, channel]) ** 2).mean(dim=1)
            errs.append(e.cpu().numpy())
        return np.concatenate(errs)


def _eval_loss(model, X, device, batch=4096):
    """Batched reconstruction loss over all channels (avoids an MPS memory spike)."""
    lossf = torch.nn.MSELoss(reduction="sum")
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], device=device)
            tot += lossf(model(xb), xb).item()
            n += xb.numel()
    return tot / n


def train_autoencoder(X, epochs=25, batch=512, lr=1e-3, val_frac=0.1,
        hidden=32, latent=16, patience=4, seed=42, device=None):
    device = device or get_device()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_frac)
    Xv, Xt = X[idx[:n_val]], X[idx[n_val:]]

    loader = DataLoader(TensorDataset(torch.as_tensor(Xt)),
                        batch_size=batch, shuffle=True)
    model = LSTMAutoencoder(n_features=X.shape[2], hidden=hidden,
                            latent=latent, seq_len=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), xb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        tr = tot / len(Xt)
        val = _eval_loss(model, Xv, device)
        print(f"epoch {ep:2d} train {tr:.4f} val {val:.4f}")
        if val < best_val - 1e-5:
            best_val = val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop (epoch {ep})")
                break
    model.load_state_dict(best_state)
    return model, best_val


def save_ae(model, best_val, name="autoencoder"):
    """persist weights + architecture config to models/<name>/."""
    out = MODELS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    cfg = {
        "n_features": model.encoder.input_size,
        "hidden": model.encoder.hidden_size,
        "latent": model.to_latent.out_features,
        "seq_len": model.seq_len,
        "window_l": WINDOW_L,
        "seasonal_period": SEASONAL_PERIOD,
        "best_val": best_val,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    return out


def load_ae(name="autoencoder", device=None):
    device=device or get_device()
    out = MODELS_DIR / name
    cfg = json.loads((out / "config.json").read_text())
    model = LSTMAutoencoder(cfg["n_features"], cfg["hidden"],
                            cfg["latent"], cfg["seq_len"]).to(device)
    model.load_state_dict(torch.load(out / "model.pt", map_location=device))
    model.eval()
    return model, cfg

